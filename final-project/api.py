import datetime
import logging
import os
from typing import List
from collections import Counter
import scipdf
import requests

import nltk
import numpy as np
import torch
import transformers
from chromadb import Settings
from langchain import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from chromadb.utils import embedding_functions
import PyPDF2
import chromadb
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer

# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.Client()
from unstructured.partition.auto import partition


class AskLLM:
    def __init__(self, model_name="gpt-3.5-turbo-0613", embedding_model="text-embedding-ada-002", temperature=0.7):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device='cuda:0')
        self.local_llm = None

    def load_local_llm(self, name='tiiuae/falcon-7b-instruct'):

        tokenizer = AutoTokenizer.from_pretrained(name)
        pipeline = transformers.pipeline(
            "text-generation",
            model=name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.tokenizer = tokenizer
        self.local_llm = pipeline

    def get_message_from_local_llm(self, input, max_length=2048, ):
        if self.local_llm is None:
            self.load_local_llm()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = self.local_llm(input,
                                 max_length=1024,
                                 max_new_tokens=max_length,
                                 do_sample=True,
                                 top_k=10,
                                 num_return_sequences=1,
                                 eos_token_id=self.tokenizer.eos_token_id,
                                 )

        return out[0]['generated_text']

    def get_message(self, input):

        current_date = datetime.date.today()
        sys_prefix_template = f"You are ChatGPT, a large language model trained by OpenAI. \n Knowledge cutoff: 2021-09 . \n Current date: {str(current_date)} "

        msg = [
            SystemMessage(content=sys_prefix_template + input["sys"]),
            HumanMessage(content=input["human"]),
        ]
        out = self.llm(msg).content
        return out

    def get_batch_message(self, input: list):
        current_date = datetime.date.today()
        sys_prefix_template = f"You are ChatGPT, a large language model trained by OpenAI. \n Knowledge cutoff: 2021-09 . \n Current date: {str(current_date)} "

        msg = []
        for i in input:
            msg.append([
                SystemMessage(content=sys_prefix_template + i["sys"]),
                HumanMessage(content=i["human"]),
            ])

        out = self.llm.generate(msg)

        output = []
        for o in out.generations:
            output.append(o[0].text)

        return output

    def get_embedding(self, input):
        if isinstance(input, list):
            return self.embeddings.embed_documents(input)
        return self.embeddings.embed_query(input)

    def get_top_n_sentences(self, doc: List[str], query, n=5):

        doc_embedding_store = self.get_embedding(doc)
        query_embedding = self.get_embedding(query)

        score = []
        for s in doc_embedding_store:
            score.append(np.dot(query_embedding, s))
        top_indices = np.argsort(score)[::-1][:n]
        return dict(
            ids=[top_indices],
            documents=[[doc[i] for i in top_indices]]
        )

    def get_top_n_sentences_by_embeddings(self, doc_embedding_store: List, doc, query, n=5, ):

        knn = NearestNeighbors(n_neighbors=n, metric="cosine")
        knn.fit(doc_embedding_store)
        query_embedding = self.get_embedding(query)

        score = []
        # for s in doc_embedding_store:
        # score.append(np.dot(query_embedding, s))
        # top_indices = np.argsort(score)[::-1][:n]
        top_indices = knn.kneighbors([query_embedding], return_distance=False)[0]
        return dict(
            ids=[top_indices],
            documents=[[doc[i] for i in top_indices]]
        )

    def get_top_n_sentences_by_chroma(self, doc: List[str], query, n=5):
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=".test_chromadb/"  # Optional, defaults to .chromadb/ in the current directory
        ))

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="text-embedding-ada-002"
        )

        # delete a collection
        client.delete_collection("all-my-documents")

        collection = client.create_collection("all-my-documents", embedding_function=openai_ef, get_or_create=True)

        collection.add(
            documents=doc,
            ids=[str(i) for i in range(len(doc))]
        )

        results = collection.query(
            query_embeddings=openai_ef(query),
            query_texts=[query],
            n_results=n
        )

        return results

    def find_most_frequent_word(self, text):
        # Tokenize the text into words
        words = text.split()

        # Count the frequency of each word
        word_counts = Counter(words)

        # Find the most frequent word
        most_frequent_word = word_counts.most_common(1)[0][0]

        return most_frequent_word

    def create_directory_to_reduce_token_length(self, text, directory_name):
        # Find the most frequent word
        most_frequent_word = self.find_most_frequent_word(text)

        # Create a directory with the most frequent word as its name
        os.mkdir(most_frequent_word)

        # Replace occurrences of the most frequent word with the directory name
        reduced_text = text.replace(most_frequent_word, directory_name)

        return reduced_text

    def ask_with_doc(self, doc, query):
        prompt_template = PromptTemplate.from_template(
            "I am going to provide you with a list of results : {documents}"
            "Instructions: Compose a comprehensive reply to the query using the search results given. "
            "Cite each reference using [ Page Number] notation (every result has this number at the beginning). "
            "Citation should be done at the end of each sentence. If the search results mention multiple subjects "
            "with the same name, create separate answers for each. Only include information found in the results and "
            "don't add any additional information. Make sure the answer is correct and don't output false content. "
            # "If the text does not relate to the query, simply state 'Text Not Found in PDF'. Ignore outlier "
            "search results which has nothing to do with the question. Only answer what is asked. The "
            "answer should be short and concise. Answer step-by-step. \n\nQuery: {query} \nAnswer your finding in detailed list: "
        )

        docs_text = ""
        for id, d in zip(doc['ids'][0], doc['documents'][0]):
            docs_text += f"Page {id}: {d} \n\n"
            pass

        prompt = prompt_template.format(
            documents=docs_text,
            query=query
        )

        print("-------------prompt---------------")
        print(prompt)

        with get_openai_callback() as cb:
            output = self.get_message(dict(
                sys="",
                human=prompt
            ))
            tokens_cost = cb.total_tokens

            return dict(
                output=output,
                tokens_cost=len(docs_text.split(" "))
            )

    def summarize_doct_by_gpt3(self, doc, min_length=100, max_length=250):
        prompt_template = PromptTemplate.from_template(
            "You need to summarize the following text: {documents}, do not exceed {max_length} words, and do not less than {min_length} words."
            "Do not modify the original text if you do not understand, and refine the english : \nSummary: "
        )

        summary = []
        for d in tqdm(doc):
            prompt = prompt_template.format(
                documents=d,
                max_length=max_length,
                min_length=min_length
            )
            summary.append(dict(
                sys="Do not reduce context length",
                human=prompt
            ))

        summary = self.get_batch_message(summary)
        print(summary)

        return summary

    def refine_doc_by_gpt3(self, doc, query):

        prompt_template = PromptTemplate.from_template(
            "You need to understand following text: {documents}, refine it based on your knowledge, and show the refined text only, and add more explain with this question {query} : \nRefined Text:"
        )

        summary = []
        for d in tqdm(doc):
            prompt = prompt_template.format(
                documents=d,
                query=query
            )
            summary.append(dict(
                sys="",
                human=prompt
            ))

        summary = self.get_batch_message(summary)
        print(summary)

        return summary

    def ask_for_the_key_referece(self, summary, abstract, ref):

        prompt_template = PromptTemplate.from_template(
            "I will give the document summary following : {summary}, and the abstract of the document is {abstract}, and the reference list as following  {ref},  please find the key reference in the document: given key reference in this format : [title of the ref , author of ref, year of ref]\n then explain why section this reference, \n And you need to explain the main research direction of the key reference: \nMain Research Direction:"
        )

        prompt = prompt_template.format(
            summary=summary,
            abstract=abstract,
            ref=ref
        )

        out = self.get_message(dict(
            sys="",
            human=prompt
        ))

        print("-------------prompt---------------")
        print(prompt)

        return dict(
            output=out,
            tokens_cost=len(prompt.split(" "))
        )

    def summarize_doc(self, doc):
        summary = []
        for d in doc:
            summary.append(self.summarizer(d, min_length=150, max_length=250, do_sample=False)[0]['summary_text'])

        logging.info(summary)

        return summary

    def summarize_doc_by_local_llm(self, doc, max_length=250):
        prompt_template = PromptTemplate.from_template(
            "You need to summarize the following text: {documents}, do not exceed {max_length} words. "
            # "Do not modify the original text if you do not understand, and refine the english if there are any wrong word,"
            " show Summary only : \nSummary: "
        )

        summary = []
        for d in tqdm(doc):
            prompt = prompt_template.format(
                documents=d,
                max_length=max_length
            )
            summary.append(self.get_message_from_local_llm(prompt, max_length + 50))

        logging.info(summary)

        return summary


class PDFReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_pdf(self):
        with open(self.file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            text = ''
            for page in range(num_pages):
                page_obj = pdf_reader.pages[page]
                text += page_obj.extract_text()
            return text

    def download_file(self, url, file_path):
        response = requests.get(url)
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print('File downloaded successfully.')

    def check_and_download_file(self):

        url = self.file_path
        assert '.pdf' in url, 'The file must be a pdf file.'
        file_path = f"file/{url.split('/')[-1]}"

        if not os.path.exists(file_path):
            self.download_file(url, file_path)
            self.file_path = file_path
        else:
            self.file_path = file_path

    def split_into_sentences(self, text):
        sentences = nltk.word_tokenize(text)
        return sentences

    def read_pdf_by_unstructured(self):
        with open(self.file_path, "rb") as f:
            elements = partition(file=f)

        text = []
        current_page = 1
        page_info = ""
        for e in elements:
            if e.metadata.page_number != current_page:
                text.append(page_info)
                current_page = e.metadata.page_number
                page_info = ""
            page_info += f"{e.text} "

        return text

    def read_pdf_by_scipdf(self):
        article_dict = scipdf.parse_pdf_to_dict(self.file_path)  # return dictionary

        text = []
        allow_keys = ['title', 'authors', 'abstract', 'sections']
        for key, value in article_dict.items():
            if key in allow_keys:
                if key == 'sections':
                    for section in value:
                        text.append(f"{section['heading']} : {section['text']}")
                elif key == 'references':
                    text.append(f"References: {value}")
                else:
                    text.append(f'{key}: {value}')

        return text
        pass

    def get_reference_list(self):

        article_dict = scipdf.parse_pdf_to_dict(self.file_path)

        ref = ""
        for key, value in article_dict.items():
            if key == 'references':
                for i, r in enumerate(value):
                    ref += f"References {i}: title : {r['title']} , author : {r['authors']}, year : {r['year']} \n"

        return ref

    def get_abstract(self):
        article_dict = scipdf.parse_pdf_to_dict(self.file_path)
        return article_dict['abstract']

    def get_pdf_title(self):
        article_dict = scipdf.parse_pdf_to_dict(self.file_path)  # return dictionary
        return article_dict['title']

    def split_by_page(self, no_references=False):
        text_list = self.read_pdf_by_unstructured()
        if no_references:
            text_list = [t for t in text_list if "references" not in t.lower()]
        return text_list

    def split_text_into_chunks(self, chunk_size):
        text = self.read_pdf_by_scipdf()
        text = " ".join(text)
        chunks = []

        # split text by sentences

        text = self.split_into_sentences(text)

        num_chars = len(text)
        start = 0
        while start < num_chars:
            end = min(start + chunk_size, num_chars)
            chunk = text[start:end]
            chunks.append(' '.join(chunk))
            start += chunk_size
        return chunks
