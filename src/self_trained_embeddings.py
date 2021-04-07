from typing import List

from gensim.models import Word2Vec
import pandas as pd
from tqdm import tqdm

from url_tokenizer import flatten_url_data, url_tokenizer


'''
Examples on how to get the self-trained embedding model.

First, get the pd.DataFrames of the dataset, e.g. dmoz_df. Then,
embedding = get_embedding(df=dmoz_df, min_count=2)
'''


def sentence_handler_func(df: pd.DataFrame):
    '''
    Cleans and extracts URLs from dataset into the input form.

    Returns:
        sentences (List[List[str]]): A list of sentences, where each sentence
            is a list of words from URL dataset.
    '''
    sentences = []
    for url in tqdm(df['url'], desc="Creating sentences"):
        try:
            sentences.append(flatten_url_data(url_tokenizer(url)))
        except AssertionError as error:
            print(f'{error} - Skipped')
    return sentences


def train_embedding_Word2Vec(sentences: List[List[str]]):
    '''
    Trains the Word2Vec model using gensim and returns the embedding.

    Args:
        sentences (List[List[str]]): A list of sentences, where each sentence
            is a list of words from URL dataset.
            e.g. [['http', 'cs-www', 'bu'], ['http', 'www', 'bu', 'edu'], ...]

    Returns:
        The trained embedding model.
    '''
    embedding = Word2Vec(sentences, min_count=2)
    return embedding


def get_embedding(df: pd.DataFrame, min_count: int = 2):
    '''
    Generate the embedding model for the choosen dataset.

    Args:
        df (pd.DataFrame):pd.DataFrames of the dataset.
        min_count (int): Ignore words that appear less than this when training
            the embedding model.

    Returns:
        The embedding model.
    '''
    sentences = sentence_handler_func(df)
    embedding = train_embedding_Word2Vec(sentences)
    return embedding
