import pandas as pd
from os import path
import re
from gensim.models import Word2Vec

from url_tokenizer import flatten_url_data,url_tokenizer
from read_data import read_dmoz, read_ilp
from util import word_splitter

'''
Examples on how to get the self-trained embedding model.

embedding_dmoz = get_embedding(dataset="dmoz", use_sample=True, min_count=2)
embedding_ilp = get_embedding(dataset="ilp", use_sample=False, min_count=2)
'''


def url_ipl_splitter(url: str,use_sample:bool=False):
    '''
    generating the urls in ilp dataset into a tuple of (protocol, domains, path, args).
    args:
        url (str): Full url of webpage.
    return:
        a tuple of (protocol, domains, path, args).
    
    remaining issue: only works on webkb, cannot process the webkb_sample dataset.
    '''
    
    # RegEx partly based on http_//cs.cornell.edu/Info/Courses/Current/CS415/CS414.html
    url_regex = re.compile(r'''
        (https?)\_\/\/                                   # http s
        ([-a-zA-Z0-9@:%._\+~#=]+\.[a-zA-Z0-9()]{1,12})  # domains
        \b
        ([-a-zA-Z0-9()@:%_\+;.~#&//=]*)                 # path
        \??
        ([-a-zA-Z0-9()@:%_\+;.~#&//=?]*)                # args
    ''', re.DOTALL | re.VERBOSE)
    match = url_regex.match(url.lower())
    assert match, f'Error matching url: {url}'
    raw_values = match.groups()
    return raw_values



def dmoz_sentence_handler(use_sample):
    '''
    clean and extract urls from dmoz dataset into the input form.
    
    return:
        sentence (list[list]):a list of sentences, where each sentence is a list of words from url dataset.
    '''

    dmoz = read_dmoz(use_sample)
    urls = list(dmoz['url'].to_numpy())

    sentence_for_train = []
    for url in urls:
        url_data = url_tokenizer(url)
        words = flatten_url_data(url_data)
        sentence_for_train.append(words)

    return sentence_for_train



def ilp_sentence_handler(use_sample):
    '''
    clean and extract urls from ilp dataset into the input form.

    return:
        sentence (list[list]):a list of sentences, where each sentence is a list of words from url dataset.
    '''
    ilp = read_ilp(use_sample)
    urls = list(ilp['url'].to_numpy())
    
    sentence_for_train = []
    for url in urls:
        url_data = []
        protocol, domains, path, args = url_ipl_splitter(url)
        if protocol is not None:
            url_data.append(protocol)
        if domains is not None:
            domains = domains.split('.')
            url_data += domains
        if path is not None:
            path = re.findall(r"[\w']+", path)
            url_data += path

        assert args == ''
        
        sentence_for_train.append(url_data)

    return sentence_for_train



def train_embedding_Word2Vec(sentence):
    '''
    train the embedding Word2Vec model using gensim.
    Args:
        sentence (list[list]):a list of sentences, where each sentence is a list of words from url dataset.
        e.g. [['http', 'cs-www', 'bu', 'edu', 'students', 'grads', 'ghuang', 'cs111', 'class', 'html'], 
        ['http', 'cs-www', 'bu', 'edu', 'students', 'grads', 'rgaimari', 'cs101', 'f96', 'cs101', 'html'],...]
    Returns:
        the trained embedding model. 

    '''
    embedding = Word2Vec(sentence,min_count=2)
    return embedding


def get_embedding(dataset,use_sample:bool=False,min_count:int=2):
    '''
    generate the embedding model for the choosen dataset.
    args:
        dataset (str): dataset's name. "dmoz" or "ilp"
        use_sample: if using the sample dataset.
        min_count (int): Ignore words that appear less than this when training the embedding model.

    return: the embedding model.
    '''

    sentence = None
    if dataset == "dmoz":
        sentence = dmoz_sentence_handler(use_sample)
    elif dataset == "ilp":
        sentence = ilp_sentence_handler(use_sample)
    
    embedding = train_embedding_Word2Vec(sentence)

    return embedding
    


