import os
from typing import Optional, Tuple

import gradio as gr
from project.api import AskLLM, PDFReader


def set_openai_api_key(api_key: str):
    """Set the api key and return chain.

    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        # chain = load_chain()
        # os.environ["OPENAI_API_KEY"] = ""
        # return chain


chat = AskLLM(model_name="gpt-3.5-turbo-0613", embedding_model="text-embedding-ada-002", temperature=0.7)


def change_title_and_prepare_doc(url: str):
    pdf_reader = PDFReader(url)
    pdf_reader.check_and_download_file()
    title = pdf_reader.get_pdf_title()

    actual_chunks = pdf_reader.split_text_into_chunks(chunk_size=200)
    # actual_chunks.extend(chat.summarize_doc(actual_chunks))
    # actual_chunks = chat.summarize_doct_by_gpt3(actual_chunks)

    doc_embeddings = chat.get_embedding(actual_chunks)

    """Change the title of the page."""
    return gr.Markdown.update(
        f'<h1><center>ChatPDF Demo, Created by <a href="https://vmv.re/in" target="_blank">Akide Liu!</a></center></h1>'
        f"<center><p>PDF Title : {title}</p></center>"
    ), actual_chunks, doc_embeddings, pdf_reader


def show_system_help_level(query, history, help_level=0):
    help_level += 1

    help_level_state = {
        1: "Entry Level assistant : I will using local summarization Model to Help you.",
        2: "Intermediate Level assistant : I will using GPT-3 to perform chunk summarization to Help you.",
        3: "Extreme Level assistant : I will call GPT-3-16k to perform multiple page level refinement and filtering, and text-davinci-003 for embedding , then GPT-4 will responsible for reasoning, the max token will be increased to 8k.",
        4: "I have tried my best to help you, but I still can't answer your question."
    }

    history = history or []

    history.append((
        "You are unsatisfied with the answer. Your Query :" + query,
        help_level_state[min(help_level, 4)]
    ))

    return history, history, help_level


def chat_with_pdf(actual_chunks, docs_embedding, query, history, pdf_reader, top_k=5, ):
    if "reference" in query.lower():
        return chat_with_key_reference(pdf_reader, query, history)

    tok_k_docs = chat.get_top_n_sentences_by_embeddings(
        docs_embedding,
        actual_chunks,
        query,
        n=top_k,
    )

    print(tok_k_docs)

    history = history or []

    output = chat.ask_with_doc(tok_k_docs, query)
    result = f'{output["output"]} \n token cost : {output["tokens_cost"]}'
    history.append((query, result))

    return history, history


def chat_with_pdf_help_level_one(pdf_reader, query, history):
    actual_chunks = pdf_reader.split_text_into_chunks(chunk_size=500)
    # actual_chunks.extend(chat.summarize_doc(actual_chunks))

    actual_chunks = chat.summarize_doc(actual_chunks)

    summary_embeddings = chat.get_embedding(actual_chunks)

    return chat_with_pdf(actual_chunks, summary_embeddings, query, history, pdf_reader, top_k=6)


def chat_with_pdf_help_level_two(pdf_reader, query, history):
    actual_chunks = pdf_reader.split_text_into_chunks(chunk_size=500)

    actual_chunks = chat.summarize_doct_by_gpt3(actual_chunks)

    summary_embeddings = chat.get_embedding(actual_chunks)

    return chat_with_pdf(actual_chunks, summary_embeddings, query, history, pdf_reader, top_k=8)


def chat_with_pdf_help_level_three(pdf_reader, query, history):
    gpt_3_super = AskLLM(model_name="gpt-3.5-turbo-16k-0613", embedding_model="text-davinci-003", temperature=0.7)

    gpt_4_reasoning = AskLLM(model_name="gpt-4-0613", temperature=0.7)

    actual_chunks = pdf_reader.split_text_into_chunks(chunk_size=1024)

    actual_chunks = gpt_3_super.refine_doc_by_gpt3(actual_chunks, query)

    summary_embeddings = gpt_3_super.get_embedding(actual_chunks)

    tok_k_docs = chat.get_top_n_sentences_by_embeddings(
        summary_embeddings,
        actual_chunks,
        query,
        n=6,
    )

    print(tok_k_docs)

    history = history or []

    output = gpt_4_reasoning.ask_with_doc(tok_k_docs, query)
    result = f'{output["output"]} \n token cost : {output["tokens_cost"]}'
    history.append((query, result))

    return history, history


def chat_with_key_reference(pdf_reader, query, history):
    summary_from_previous = history[-1][1]

    gpt_4_reasoning = AskLLM(model_name="gpt-3.5-turbo-16k-0613", temperature=0.7)

    abstract = pdf_reader.get_abstract()

    reference = pdf_reader.get_reference_list()

    output = gpt_4_reasoning.ask_for_the_key_referece(
        summary_from_previous,
        abstract,
        reference,
    )

    result = f'{output["output"]} \n token cost : {output["tokens_cost"]}'

    history.append((query, result))

    return history, history


def chat_with_help(pdf_reader, query, history, help_level):
    if int(help_level) == 1:
        return chat_with_pdf_help_level_one(pdf_reader, query, history)
    elif int(help_level) == 2:
        return chat_with_pdf_help_level_two(pdf_reader, query, history)
    elif int(help_level) == 3:
        return chat_with_pdf_help_level_three(pdf_reader, query, history)
    return history, history


block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        title = gr.Markdown(
            '<h1><center>ChatPDF Demo, Created by <a href="https://vmv.re/in" target="_blank">Akide Liu!</a></center></h1>')

        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key (sk-...)",
            show_label=False,
            lines=1,
            type="password",
        )

        pdf_link = gr.Textbox(
            placeholder="Paste your PDF link",
            show_label=False,
            lines=1,
            type="text",
        )

    chatbot = gr.Chatbot(height=700)

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="What's the answer to life, the universe, and everything?",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)
        ask_for_help = gr.Button(value="Help!", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "What's title of this PDF?",
            "What is the hypothesis about alignment in this paper?",
            "What is the main discovery of this paper?",
            "find the key reference for the following paper?",
        ],
        inputs=message,
    )

    state = gr.State()
    agent_state = gr.State()
    document_state = gr.State()
    document_embeddings = gr.State()
    help_level = gr.State(value=0)
    pdf_reader_class = gr.State()

    submit.click(chat_with_pdf, inputs=[document_state, document_embeddings, message, state, pdf_reader_class],
                 outputs=[chatbot, state])
    message.submit(chat_with_pdf, inputs=[document_state, document_embeddings, message, state, pdf_reader_class],
                   outputs=[chatbot, state])

    ask_for_help.click(show_system_help_level, inputs=[message, state, help_level],
                       outputs=[chatbot, state, help_level]).then(
        chat_with_help, inputs=[pdf_reader_class, message, state, help_level], outputs=[chatbot, state]
    )

    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox],
        outputs=[agent_state],
    )

    pdf_link.change(
        change_title_and_prepare_doc,
        inputs=[pdf_link],
        outputs=[title, document_state, document_embeddings, pdf_reader_class],
    )

    # reset help level to 0
    message.change(
        lambda level: 0,
        inputs=[help_level],
        outputs=[help_level],
    )

block.launch(debug=True)
