import nltk
import pandas as pd
from matplotlib import pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
import sys
from multiprocessing import Pool
from typing import Iterable
from tqdm.notebook import tqdm
from nltk.stem import PorterStemmer, SnowballStemmer  # Porter's algorithm is the most common algorithm for stemming English
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from easydict import EasyDict as edict, EasyDict
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
import seaborn as sns
from datasets import Dataset
from transformers import DataCollatorWithPadding
import evaluate
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
import pprint
import spacy
from spacy import displacy
from textblob import TextBlob
import torch
import torch.nn.functional as F



import xml.etree.ElementTree as ET

sns.set()

# check nltk exist
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('all')


# multi thread processing and progress bar
# ref : https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/progressbar.py
def track_parallel_progress(func,
                            tasks,
                            nproc,
                            initializer=None,
                            initargs=None,
                            bar_width=50,
                            chunksize=1,
                            skip_first=False,
                            keep_order=True,
                            file=sys.stdout):
    """Track the progress of parallel task execution with a progress bar.
    The built-in :mod:`multiprocessing` module is used for process pools and
    tasks are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.
    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        nproc (int): Process (worker) number.
        initializer (None or callable): Refer to :class:`multiprocessing.Pool`
            for details.
        initargs (None or tuple): Refer to :class:`multiprocessing.Pool` for
            details.
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
        bar_width (int): Width of progress bar.
        skip_first (bool): Whether to skip the first sample for each worker
            when estimating fps, since the initialization step may takes
            longer.
        keep_order (bool): If True, :func:`Pool.imap` is used, otherwise
            :func:`Pool.imap_unordered` is used.
    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    pool = init_pool(nproc, initializer, initargs)
    start = not skip_first
    task_num -= nproc * chunksize * int(skip_first)
    # prog_bar = ProgressBar(task_num, bar_width, start, file=file)
    results = []
    if keep_order:
        gen = pool.imap(func, tasks, chunksize)
    else:
        gen = pool.imap_unordered(func, tasks, chunksize)
    for i, result in tqdm(enumerate(gen), total=len(tasks)):
        results.append(result)
        if skip_first:
            if len(results) < nproc * chunksize:
                continue
            elif len(results) == nproc * chunksize:
                # prog_bar.start()
                continue
        # prog_bar.update()
    # prog_bar.file.write('\n')
    pool.close()
    pool.join()
    return results


def init_pool(process_num, initializer=None, initargs=None):
    if initializer is None:
        return Pool(process_num)
    elif initargs is None:
        return Pool(process_num, initializer)
    else:
        if not isinstance(initargs, tuple):
            raise TypeError('"initargs" must be a tuple')
        return Pool(process_num, initializer, initargs)




def load_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root

# xml_data = load_xml('path_to_your_xml_file.xml')

def get_xml_structure(element, indent=''):
    print(indent + element.tag)  # Print the tag name of the current element
    for child in element:
        get_xml_structure(child, indent + '  ')  # Recursively call the function for each child element


def parse_xml_to_df(root):
    # Initialize a list to store the data
    parsed_data = []

    for sentence in root.findall('sentence'):
        sentence_data = {
            "id": sentence.get('id'),
            "text": sentence.find('text').text,
            "aspect_terms": [],
            "aspect_categories": []
        }

        aspect_terms = sentence.find('aspectTerms')
        # sentence.find('text')

        if aspect_terms is not None:
            for term in aspect_terms.findall('aspectTerm'):
                term_details = {
                    "term": term.get('term'),
                    "polarity": term.get('polarity'),
                    "from": term.get('from'),
                    "to": term.get('to')
                }
                sentence_data["aspect_terms"].append(term_details)

        aspect_categories = sentence.find('aspectCategories')
        if aspect_categories is not None:
            for category in aspect_categories.findall('aspectCategory'):
                category_details = {
                    "category": category.get('category'),
                    "polarity": category.get('polarity')
                }
                sentence_data["aspect_categories"].append(category_details)

        sentence_data = EasyDict(sentence_data)
        # Add the sentence data to the list
        parsed_data.append(sentence_data)


    return parsed_data