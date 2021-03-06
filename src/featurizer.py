import csv
import re
from time import time
import os
from os.path import dirname, abspath
from typing import Dict, List, Tuple, Union

from util import flatten, flatten_twice
from url_tokenizer import url_tokenizer, UrlData, flatten_url_data, \
                          url_raw_splitter, url_html_decoder

import numpy as np
import pandas as pd
import gensim.downloader as api
from gensim.models.fasttext import FastTextKeyedVectors

EmbeddingIndex = Union[Dict[str, List[float]], FastTextKeyedVectors]
VectorMatrix = Tuple[np.ndarray, np.ndarray]

# Hyperparameters that can be varied
SUB_DOMAIN_DEFAULT_MAX_LEN = 5
MAIN_DOMAIN_DEFAULT_MAX_LEN = 5
PATH_DEFAULT_MAX_LEN = 10
ARG_DEFAULT_MAX_LEN = 10
UNTRUSTWORTHY_TLDS = set([
    'ga', 'tk', 'ml', 'cf', 'surf', 'su', 'ba', 'cyou', 'support', 'bd', 'th',
    'casa', 'pk', 'top', 'id', 'link', 'sa', 'in', 'xyz', 'pw', 'monster',
    'ink' 'lk', 'wang', 'cc', 'vn', 'np', 'ec', 'tr', 'shop', 'ge', 'pe',
    'ke', 'xn--p1ai', 'buzz', 'ng', 'ir', 'me', 'ma', 'digital', 'at', 'live',
    'club', 'services', 'icu', 'cl', 'it', 'pl', 'cam', 'my', 'ru', 'today',
    'ae', 'sg'
])

# These following constants should not be changed
GLOVE, CONCEPTNET, WORD2VEC, FASTTEXT, SAMPLE = \
    'GloVe', 'Conceptnet', 'Word2Vec', 'FastText', 'sample'

WORD_EMBED_TO_GENSIM_FILE = {
    GLOVE: 'glove-wiki-gigaword-300',
    CONCEPTNET: 'conceptnet-numberbatch-17-06-300',
    WORD2VEC: 'word2vec-google-news-300',
    FASTTEXT: 'fasttext-wiki-news-subwords-300',
}

CUR_DIR = dirname(abspath(__file__))
WORD_EMBED_PATH = os.path.join(CUR_DIR, 'word_embed')
SAMPLE_FILE = os.path.join(WORD_EMBED_PATH, SAMPLE, 'sample.txt')


class UrlFeaturizer:
    def __init__(self,
                 embedding: Union[str, FastTextKeyedVectors] = CONCEPTNET,
                 expand_tokens: bool = False,
                 sub_domain_max_len: int = SUB_DOMAIN_DEFAULT_MAX_LEN,
                 main_domain_max_len: int = MAIN_DOMAIN_DEFAULT_MAX_LEN,
                 path_max_len: int = PATH_DEFAULT_MAX_LEN,
                 arg_max_len: int = ARG_DEFAULT_MAX_LEN,
                 is_sequential: bool = False,
                 verbose: bool = True):
        '''
        Returns a new UrlFeaturizer with the loaded word embedding and settings

        Args:
            embedding (Union[str, FastTextKeyedVectors]): The embedding to use,
                i.e. a string like 'GloVe' or 'Conceptnet' or a FastText model
            expand_tokens (bool): Whether or not to perform word expansion
            sub_domain_max_len (int): The length of the sub_domain matrix in
                the generated word matrix
            main_domain_max_len (int): The length of the main_domain matrix in
                the generated word matrix
            path_max_len (int): The length of the path matrix in the generated
                word matrix
            arg_max_len (int): The length of the args matrix in the generated
                word matrix
            is_sequential (bool): Whether to use sequential embedding or
                positional embedding
            verbose (bool): Whether or not to print logging messages

        Returns:
            url_featurizer (UrlFeaturizer): UrlFeaturizer instance
        '''
        t_start = time()
        self.verbose = verbose
        self.__create_word_matrix__ = (
            self.__create_sequential_word_matrix__ if is_sequential
            else self.__create_positional_word_matrix__)
        self.expand_tokens = expand_tokens
        self.embed_prefix = ''
        self.sub_domain_max_len = sub_domain_max_len
        self.main_domain_max_len = main_domain_max_len
        self.path_max_len = path_max_len
        self.arg_max_len = arg_max_len
        self.N = self.__calc_n__()
        self.is_fasttext = isinstance(embedding, FastTextKeyedVectors)
        self.embeddings_index = self.__read_embeddings__(embedding)
        self.embedding_dim = len(self.__get_embed__('the'))
        self.avg_vec = self.__create_avg_vec__(embedding)
        self.hand_picked_feat_len = self.__hand_picked_feat_len__()
        if self.verbose:
            elapsed = time() - t_start
            embed_str = 'FastText' if self.is_fasttext else embedding
            print(f'Created {embed_str} UrlFeaturizer in {elapsed:.1f} s')

    def __calc_n__(self):
        '''
        Calculates the length of the word embedding N

        Returns:
            N (int): Length of word embedding
        '''
        N = (self.sub_domain_max_len + self.main_domain_max_len + 1
             + self.path_max_len + self.arg_max_len)
        return N

    def __read_embeddings__(self,
                            embedding: Union[str, FastTextKeyedVectors]) \
            -> EmbeddingIndex:
        '''
        Takes the choice of embedding and returns a dictionary with the word
        as key and the word embedding as the value

        Args:
            embedding (Union[str, FastTextKeyedVectors]): The embedding to use,
                i.e. a string like 'GloVe' or 'Conceptnet' or a FastText model

        Returns:
            embedding (EmbeddingIndex): A string-vector dictionary of the
                embeddings
        '''
        if self.is_fasttext:
            return embedding
        elif embedding in WORD_EMBED_TO_GENSIM_FILE:
            return self.__read_gensim_embeddings__(embedding)
        elif embedding == SAMPLE:
            return self.__read_sample_embeddings__()
        else:
            valid_choices = list(WORD_EMBED_TO_GENSIM_FILE.keys())
            valid_choices.append(SAMPLE)
            msg = (f'"{embedding}" is not a valid embedding choice. Try one ' +
                   f'of {valid_choices} or a FastTextKeyedVectors model')
            raise ValueError(msg)

    def __read_gensim_embeddings__(self, embedding: str) -> EmbeddingIndex:
        '''
        Takes the choice of embedding and returns a dictionary with the word
        as key and the word embedding as the value

        Args:
            embedding (str): String, should be one of the keys in
                             WORD_EMBED_TO_GENSIM_FILE.

        Returns:
            embedding (EmbeddingIndex): A string-vector dictionary of the
                                        embedding
        '''
        if embedding == CONCEPTNET:
            self.embed_prefix = '/c/en/'

        embedding_file = WORD_EMBED_TO_GENSIM_FILE[embedding]
        if self.verbose:
            print(f'Reading the {embedding_file} word vector file...')
        embeddings = api.load(embedding_file)
        return embeddings

    def __read_sample_embeddings__(self) -> EmbeddingIndex:
        '''
        Reads the sample embeddings and returns a dictionary of these

        Returns:
            embedding (EmbeddingIndex): A string-vector dictionary of the
                                        sample embedding
        '''
        if self.verbose:
            print('Reading the sample word vector file...')

        words_df = pd.read_csv(SAMPLE_FILE, sep=" ", index_col=0,
                               na_values=None, keep_default_na=False,
                               header=None, quoting=csv.QUOTE_NONE)
        if self.verbose:
            print('Creating dictionary index...')
        embeddings_index = {word: words_df.loc[word].values
                            for word in words_df.index.values}
        return embeddings_index

    def __create_avg_vec__(self, embedding: Union[str, FastTextKeyedVectors]) \
            -> np.ndarray:
        '''
        Creates a vector that is the average of all word vectors

        Returns:
            avg_vec (np.ndarray): Average word vector
        '''
        if self.verbose:
            print('Creating the average vector of all the word vectors...')

        if self.is_fasttext:  # FastText handles OOV itself
            return np.zeros(self.embedding_dim)

        word_matrix = (list(self.embeddings_index.values())
                       if embedding == SAMPLE
                       else self.embeddings_index.vectors)
        avg_vec = np.mean(np.array(word_matrix), axis=0)
        return avg_vec

    def __hand_picked_feat_len__(self):
        '''
        Returns an integer denoting how long the hand-picked feature vector is

        Returns:
            feat_len (int): Length of hand-picked feature vector
        '''
        sample_url = 'http://test.com'
        sample_url_data = url_tokenizer(sample_url)
        feat_len = len(self.__create_hand_picked_features__(sample_url,
                                                            sample_url_data))
        return feat_len

    def __get_embed__(self, token: str) -> np.ndarray:
        '''
        Takes a single token and returns the corresponding embedding. If token
        is not in vocabulary, the average word vector is returned

        Args:
            token (str): Token to get embedding for

        Returns:
            word_embed (np.ndarray): Word embedding array for token if present,
                                     otherwise average word embedding array
        '''
        p_word = self.embed_prefix + token
        word_embed = (self.embeddings_index[p_word]
                      if p_word in self.embeddings_index
                      else self.avg_vec)
        return word_embed

    def __word_embed__(self, tokens: List[str], mat_len: int) -> np.ndarray:
        '''
        Takes a list of tokens and the length of the matrix and creates a word
        embedding from this of shape (mat_len, self.embedding_dim)

        Args:
            tokens (str): A list of tokenized words
            mat_len (int): The desired length of the matrix

        Returns:
            embed_matrix (np.ndarray): Word embedding submatrix of
                                       shape (mat_len, embedding_dim)
        '''
        embed_matrix = np.zeros((mat_len, self.embedding_dim))
        for i, token in enumerate(tokens[:mat_len]):
            embed_matrix[i] = self.__get_embed__(token)
        return embed_matrix

    def __create_positional_word_matrix__(self, url_data: UrlData) \
            -> np.ndarray:
        '''
        Takes the url_date and creates a full word embedding from this of shape
        (N, embedding_dim), where N is equal to the sum of the lengths of
        the sub_domains, main_domains, paths, args + 1 for the TLD. The
        embedding will be positional, meaning it will have the embed vectors
        for each area of the URL starting at fixed locations and truncated
        to fixed lengths. I.e. [0, 0, x, y, 0, 0, d, f, 0, ...]

        Args:
            url_data (UrlData): 4-tuple of url data

        Returns:
            word_matrix (np.ndarray): Full positional word embedding matrix of
                                      shape (N, embedding_dim)
        '''
        _, domains, path, args = url_data
        sub_domains, main_domain, domain_ending = domains
        args_flat = flatten_twice(args)

        sub_domain_mat = self.__word_embed__(sub_domains, self.sub_domain_max_len)
        main_domain_mat = self.__word_embed__(main_domain, self.main_domain_max_len)
        domain_end_vec = self.__word_embed__([domain_ending], 1)
        path_mat = self.__word_embed__(path, self.path_max_len)
        args_mat = self.__word_embed__(args_flat, self.arg_max_len)
        word_matrix = np.concatenate([
            sub_domain_mat, main_domain_mat,
            domain_end_vec,
            path_mat, args_mat
        ])
        return word_matrix

    def __create_sequential_word_matrix__(self, url_data: UrlData) \
            -> np.ndarray:
        '''
        Takes the url_date and creates a full word embedding from this of shape
        (N, embedding_dim), where N is equal to the sum of the lengths of
        the sub_domains, main_domains, paths, args + 1 for the TLD. The
        embedding will be sequential, meaning it will have the embed vectors
        for each word at the beginning and then padded/truncated to fixed
        length N. I.e. [x, y, z, k, d, v, f, 0, 0, 0, ...]

        Args:
            url_data (UrlData): 4-tuple of url data

        Returns:
            word_matrix (np.ndarray): Full sequential word embedding matrix of
                                      shape (N, embedding_dim)
        '''
        _, domains, path, args = url_data
        sub_domains, main_domain, domain_ending = domains
        args_flat = flatten_twice(args)
        full = sub_domains + main_domain + [domain_ending] + path + args_flat
        word_matrix = self.__word_embed__(full, self.N)
        return word_matrix

    def __create_hand_picked_features__(self, url: str,
                                        url_data: UrlData) -> np.ndarray:
        '''
        Creates hand-picked features based on the url data

        Args:
            url (str): URL string
            url_data (UrlData): 4-tuple of URL data

        Returns:
            feat_vec (np.ndarray): 1D vector of hand-picked features
        '''
        words = flatten_url_data(url_data)

        url_decoded = url_html_decoder(url)
        protocol, domains_raw, path_raw, args_raw = \
            url_raw_splitter(url_decoded)

        domain_len = len(domains_raw)
        path_len = len(path_raw)
        args_len = len(args_raw)

        dot_count_in_path_and_args = path_raw.count('.') + args_raw.count('.')
        capital_count = len([char for char in url_decoded if char.isupper()])

        domain_is_ip_address = int(bool(re.match(r'(\d+\.){3}\d+',
                                                 domains_raw)))
        contain_suspicious_symbol = int(args_raw.find('\\') >= 0 or
                                        args_raw.find(':') >= 0)

        protocol, domains, path, args = url_data
        sub_domains, main_domain, domain_ending = domains

        contains_at_symbol = int(len(path) > 0 and path[-1] == '@')
        is_https = int(protocol == 'https')
        num_main_domain_words = len(main_domain)
        num_sub_domains = len(sub_domains)
        is_www = int(num_sub_domains > 0 and sub_domains[0] == 'www')
        is_www_weird = int(num_sub_domains > 0 and
                           bool(re.match(r'www.+', sub_domains[0])))
        num_path_words = len(path) - contains_at_symbol
        domain_end_verdict = int(domain_ending in UNTRUSTWORTHY_TLDS)

        sub_domain_chars = flatten(sub_domains)
        sub_domains_num_digits = len([char for char in sub_domain_chars
                                      if char.isdigit()])

        path_chars = flatten(path)
        path_num_digits = len([char for char in path_chars if char.isdigit()])

        args_flat = flatten_twice(args)
        args_chars = flatten(args_flat)
        args_num_digits = len([char for char in args_chars if char.isdigit()])

        total_num_digits = (sub_domains_num_digits
                            + path_num_digits
                            + args_num_digits)

        word_court_in_url = len(words) - contains_at_symbol

        feat_vec = np.array([
            is_https, num_main_domain_words, num_sub_domains,
            is_www, is_www_weird, num_path_words, domain_end_verdict,
            sub_domains_num_digits, path_num_digits, args_num_digits,
            total_num_digits, contains_at_symbol, word_court_in_url,
            domain_len, path_len, args_len,
            dot_count_in_path_and_args, capital_count,
            domain_is_ip_address, contain_suspicious_symbol
        ])
        return feat_vec

    def __featurize__(self, url: str) -> VectorMatrix:
        '''
        Takes a single url and returns a vector of hand picked features as
        well as the word embedding matrix

        Args:
            url (str): URL string

        Returns:
            feat_vec (np.ndarray): Hand-picked features vector
            word_matrix (np.ndarray): Full word embedding matrix of shape
                                      (N, embedding_dim)
        '''
        try:
            url_data = url_tokenizer(url, expand_tokens=self.expand_tokens)
            feat_vec = self.__create_hand_picked_features__(url, url_data)
            word_matrix = self.__create_word_matrix__(url_data)
            return feat_vec, word_matrix
        except Exception as e:
            print(f'Error with "{url}": {e}')
            feat_vec_placeholder = np.zeros(self.hand_picked_feat_len)
            word_matrix_placeholder = np.zeros((self.N, self.embedding_dim))
            return feat_vec_placeholder, word_matrix_placeholder

    def featurize(self, urls: Union[str, List[str]]) \
            -> Union[VectorMatrix, List[VectorMatrix]]:
        '''
        Takes either a single URL or a list of URLs and return respectively a
        (features_vec, word_matrix) tuple or list of those tuples for each URL.

        Args:
            urls (Union[str, List[str]]): URL string or list of URL strings

        Returns:
            feat_vec_word_mat (Union[VectorMatrix, List[VectorMatrix]]): Either
                tuple of (features_vec, word_matrix) if a single URL is
                supplied or a list of these tuples if list of URLS is supplied
        '''
        if isinstance(urls, str):
            feat_vec_word_mat = self.__featurize__(urls)
        else:  # list of urls
            feat_vec_word_mat = [self.__featurize__(url) for url in urls]
        return feat_vec_word_mat

    def set_hyperparams(self,
                        sub_domain_max_len: Union[int, None] = None,
                        main_domain_max_len: Union[int, None] = None,
                        path_max_len: Union[int, None] = None,
                        arg_max_len: Union[int, None] = None):
        '''
        Allows resetting the hyperparams for the lengths of differnt values
        after initialization. This is to allow for more experimentation given
        that initialization is slow due to reading a large word embedding

        Args:
            sub_domain_max_len (Union[int, None]): sub_domain_max_len
            main_domain_max_len (Union[int, None]): main_domain_max_len
            path_max_len (Union[int, None]): path_max_len
            arg_max_len (Union[int, None]): arg_max_len
        '''
        if sub_domain_max_len:
            self.sub_domain_max_len = sub_domain_max_len
        if main_domain_max_len:
            self.main_domain_max_len = main_domain_max_len
        if path_max_len:
            self.path_max_len = path_max_len
        if arg_max_len:
            self.arg_max_len = arg_max_len
        self.N = self.__calc_n__()
