import re
from typing import List

from gensim.models import Word2Vec

from url_tokenizer import flatten_url_data, url_tokenizer
from read_data import read_dmoz, read_ilp

import pandas as pd


'''
Examples on how to get the self-trained embedding model.

First, get the pd.DataFrames of the dataset, e.g. dmoz_df. Then,
embedding = get_embedding(df=dmoz_df,use_sample=True,min_count=2)

'''


def sentence_handler_func(df: pd.DataFrame,use_sample: bool = False):
    '''
    Cleans and extracts URLs from dataset into the input form.

    Returns:
        sentences (List[List[str]]): A list of sentences, where each sentence
            is a list of words from URL dataset.
    '''
    
    urls = df['url'].to_numpy()
    sentences_for_train = [flatten_url_data(url_tokenizer(url))
                           for url in urls]
    return sentences_for_train



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


def get_embedding(df: pd.DataFrame, use_sample: bool = False, min_count: int = 2):
    '''
    Generate the embedding model for the choosen dataset.

    Args:
        df (pd.DataFrame):pd.DataFrames of the dataset.
        use_sample (bool): Whether to use sample dataset.
        min_count (int): Ignore words that appear less than this when training
            the embedding model.

    Returns:
        The embedding model.
    '''
    
    sentences = sentence_handler_func(df,use_sample)
    embedding = train_embedding_Word2Vec(sentences)
    return embedding
