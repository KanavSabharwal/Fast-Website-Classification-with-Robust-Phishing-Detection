import re
from typing import List

from gensim.models import Word2Vec

from url_tokenizer import flatten_url_data, url_tokenizer
from read_data import read_dmoz, read_ilp

'''
Examples on how to get the self-trained embedding model.

embedding_dmoz = get_embedding(dataset="dmoz", use_sample=True, min_count=2)
embedding_ilp = get_embedding(dataset="ilp", use_sample=False, min_count=2)
'''

DMOZ, ILP = 'DMOZ', 'ILP'


def url_ilp_splitter(url: str, use_sample: bool = False):
    '''
    Transforms the urls in ilp dataset into a tuple of
    (protocol, domains, path, args).

    Args:
        url (str): Full url of webpage.

    Returns:
        A tuple of (protocol, domains, path, args).
    '''
    # TODO: Only works on webkb, cannot process the webkb_sample dataset.

    # RegEx partly based on
    # http_//cs.cornell.edu/Info/Courses/Current/CS415/CS414.html
    url_regex = re.compile(r'''
        (https?)\_\/\/                                   # http s
        ([-a-zA-Z0-9@:%._\+~#=]+\.[a-zA-Z0-9()]{1,12})   # domains
        \b
        ([-a-zA-Z0-9()@:%_\+;.~#&//=]*)                  # path
        \??
        ([-a-zA-Z0-9()@:%_\+;.~#&//=?]*)                 # args
    ''', re.DOTALL | re.VERBOSE)
    match = url_regex.match(url.lower())
    assert match, f'Error matching url: {url}'
    raw_values = match.groups()
    return raw_values


def dmoz_sentence_handler(use_sample: bool = False):
    '''
    Cleans and extracts URLs from DMOZ dataset into the input form.

    Returns:
        sentences (List[List[str]]): A list of sentences, where each sentence
            is a list of words from URL dataset.
    '''
    dmoz = read_dmoz(use_sample)
    urls = dmoz['url'].to_numpy()
    sentences_for_train = [flatten_url_data(url_tokenizer(url))
                           for url in urls]
    return sentences_for_train


def ilp_sentence_handler(use_sample: bool = False):
    '''
    Clean and extracts urls from ILP dataset into the input form.

    Returns:
        sentences (List[List[str]]): A list of sentences, where each sentence
            is a list of words from URL dataset.
    '''
    ilp = read_ilp(use_sample)
    urls = ilp['url'].to_numpy()

    sentences_for_train = []
    for url in urls:
        url_data = []
        protocol, domains, path, args = url_ilp_splitter(url)
        if protocol is not None:
            url_data.append(protocol)
        if domains is not None:
            domains = domains.split('.')
            url_data += domains
        if path is not None:
            path = re.findall(r"[\w']+", path)
            url_data += path

        sentences_for_train.append(url_data)

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


def get_embedding(dataset: str, use_sample: bool = False, min_count: int = 2):
    '''
    Generate the embedding model for the choosen dataset.

    Args:
        dataset (str): Dataset's name. "dmoz" or "ilp"
        use_sample (bool): Whether to use sample dataset.
        min_count (int): Ignore words that appear less than this when training
            the embedding model.

    Returns:
        The embedding model.
    '''
    sentence_handler_func = {
        DMOZ: dmoz_sentence_handler,
        ILP: ilp_sentence_handler
    }
    assert dataset in sentence_handler_func, \
        f'Only {list(sentence_handler_func.keys())} supported.'
    sentences = sentence_handler_func[dataset](use_sample)
    embedding = train_embedding_Word2Vec(sentences)
    return embedding
