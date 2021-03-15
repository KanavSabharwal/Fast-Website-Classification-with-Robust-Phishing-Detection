import csv
import re
from time import time
import os
from os.path import dirname, abspath
from typing import Dict, List, Tuple, Union

from util import flatten_twice
from url_tokenizer import url_tokenizer, UrlData

import numpy as np
import pandas as pd

EmbeddingIndex = Dict[str, List[float]]
VectorMatrix = Tuple[np.ndarray, np.ndarray]

# Hyperparameters that can be varied
SUB_DOMAIN_DEFAULT_MAX_LEN = 5
MAIN_DOMAIN_DEFAULT_MAX_LEN = 5
PATH_DEFAULT_MAX_LEN = 10
ARG_DEFAULT_MAX_LEN = 10
TRUSTWORTHY_TLDS = set(['com', 'net', 'edu', 'org', 'gov'])
UNTRUSTWORTHY_TLDS = set(['xyz', 'biz', 'info'])

# These values should not be changed
GLOVE, CONCEPTNET = 'glove', 'conceptnet'
CUR_DIR = dirname(abspath(__file__))
WORD_EMBED_PATH = os.path.join(CUR_DIR, 'word_embed')
GLOVE_FILE = os.path.join(WORD_EMBED_PATH, GLOVE, 'glove.42B.300d.txt')
CONCEPTNET_FILE = os.path.join(WORD_EMBED_PATH, CONCEPTNET, 'numberbatch-19.08.txt')


class UrlFeaturizer:
    def __init__(self,
                 embedding: str = CONCEPTNET,
                 sub_domain_max_len: int = SUB_DOMAIN_DEFAULT_MAX_LEN,
                 main_domain_max_len: int = MAIN_DOMAIN_DEFAULT_MAX_LEN,
                 path_max_len: int = PATH_DEFAULT_MAX_LEN,
                 arg_max_len: int = ARG_DEFAULT_MAX_LEN,
                 verbose: bool = True):
        '''
        Returns a new UrlFeaturizer with the loaded word embedding and settings

        Args:
            embedding (str): A list of tokenized words
            sub_domain_max_len (int): The length of the sub_domain matrix in
                                      the generated word matrix
            main_domain_max_len (int): The length of the main_domain matrix in
                                       the generated word matrix
            path_max_len (int): The length of the path matrix in the generated
                                word matrix
            arg_max_len (int): The length of the args matrix in the generated
                               word matrix
            verbose (bool): Whether or not to print logging messages

        Returns:
            url_featurizer (UrlFeaturizer): UrlFeaturizer instance
        '''
        t_start = time()
        self.verbose = verbose
        self.sub_domain_max_len = sub_domain_max_len
        self.main_domain_max_len = main_domain_max_len
        self.path_max_len = path_max_len
        self.arg_max_len = arg_max_len
        self.N = self.__calc_n__()
        self.embeddings_index = self.__read_embeddings__(embedding)
        self.embedding_dim = len(self.embeddings_index['the'])
        self.unknown_vec = self.__create_unknown_vec__()
        self.hand_picked_feat_len = self.__hand_picked_feat_len__()
        if self.verbose:
            elapsed = time() - t_start
            print(f'Created {embedding} UrlFeaturizer in {elapsed:.1f} s')

    def __calc_n__(self):
        '''
        Calculates the length of the word embedding N

        Returns:
            N (int): Length of word embedding
        '''
        N = self.sub_domain_max_len + self.main_domain_max_len + 1 + \
            self.path_max_len + self.arg_max_len
        return N

    def __read_embeddings__(self, embedding: str) -> EmbeddingIndex:
        '''
        Takes the choice of embedding and returns a dictionary with the word
        as key and the word embedding as the value

        Args:
            embedding (str): String, should be "glove" or "conceptnet"

        Returns:
            embedding (EmbeddingIndex): A string-vector dictionary of the
                                        embedding
        '''
        if embedding == CONCEPTNET:
            return self.__read_conceptnet_embeddings__()
        elif embedding == GLOVE:
            return self.__read_glove_embeddings__()
        else:
            msg = f'{embedding} is not a valid embedding choice. ' + \
                  f'Try one of {[CONCEPTNET, GLOVE]}'
            raise ValueError(msg)

    def __read_conceptnet_embeddings__(self) -> EmbeddingIndex:
        '''
        Reads the ConceptNet embeddings and returns a dictionary of these

        Returns:
            embedding (EmbeddingIndex): A string-vector dictionary of the
                                        ConceptNet embedding
        '''
        if self.verbose:
            print('Reading the ConceptNet word vector file...')
        words_df = pd.read_csv(CONCEPTNET_FILE, sep=" ", index_col=0,
                               skiprows=1, header=None,
                               keep_default_na=False, na_values=None,
                               quoting=csv.QUOTE_NONE)
        if self.verbose:
            print('Remapping word values...')
        words_df.index = words_df.index.map(lambda x: x.split('/')[-1])
        if self.verbose:
            print('Creating dictionary index...')
        embeddings_index = {word: words_df.loc[word].values
                            for word in words_df.index.values}
        return embeddings_index

    def __read_glove_embeddings__(self) -> EmbeddingIndex:
        '''
        Reads the GloVe embeddings and returns a dictionary of these

        Returns:
            embedding (EmbeddingIndex): A string-vector dictionary of the
                                        GloVe embedding
        '''
        if self.verbose:
            print('Reading the GloVe word vector file...')
        words_df = pd.read_csv(GLOVE_FILE, sep=" ", index_col=0,
                               na_values=None, keep_default_na=False,
                               header=None, quoting=csv.QUOTE_NONE)
        if self.verbose:
            print('Creating dictionary index...')
        embeddings_index = {word: words_df.loc[word].values
                            for word in words_df.index.values}
        return embeddings_index

    def __create_unknown_vec__(self) -> np.ndarray:
        '''
        Creates a vector that is the average of all word vectors

        Returns:
            unknown_vec (np.ndarray): Average word vector
        '''
        if self.verbose:
            print('Creating the average vector of all the word vectors')
        word_embed_vector_lst = list(self.embeddings_index.values())
        unknown_vec = np.mean(word_embed_vector_lst, axis=0)
        return unknown_vec

    def __hand_picked_feat_len__(self):
        '''
        Returns an integer denoting how long the hand-picked feature vector is

        Returns:
            feat_len (int): Length of hand-picked feature vector
        '''
        sample_url_data = url_tokenizer('http://test.com')
        feat_len = len(self.__create_hand_picked_features__(sample_url_data))
        return feat_len

    def __word_embed__(self, word_lst: List[str], mat_len: int) -> np.ndarray:
        '''
        Takes a list of words and the length of the matrix and creates a word
        embedding from this of shape (mat_len, self.embedding_dim)

        Args:
            word_lst (str): A list of tokenized words
            mat_len (int): The desired length of the matrix

        Returns:
            embed_matrix (np.ndarray): Word embedding submatrix of
                                       shape (mat_len, embedding_dim)
        '''
        embed_matrix = np.zeros((mat_len, self.embedding_dim))
        for i, word in enumerate(word_lst[:mat_len]):
            in_vocab = word in self.embeddings_index
            vec = self.embeddings_index[word] if in_vocab else self.unknown_vec
            embed_matrix[i] = vec
        return embed_matrix

    def __create_word_matrix__(self, url_data: UrlData) -> np.ndarray:
        '''
        Takes the url_date and creates a full word embedding from this of shape
        (N, embedding_dim), where N is equal to the sum of the lengths of
        the sub_domains, main_domains, paths, args + 1 for the TLD

        Args:
            url_data (UrlData): 4-tuple of url data

        Returns:
            word_matrix (np.ndarray): Full word embedding matrix of
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

    def __create_hand_picked_features__(self, url_data: UrlData) -> np.ndarray:
        '''
        Creates hand-picked features based on the url data

        Args:
            url_data (UrlData): 4-tuple of url data

        Returns:
            feat_vec (np.ndarray): 1D vector of hand-picked features
        '''
        protocol, domains, path, args = url_data
        sub_domains, main_domain, domain_ending = domains

        is_https = int(protocol == 'https')
        num_main_domain_words = len(main_domain)
        num_sub_domains = len(sub_domains)
        is_www = int(num_sub_domains > 0 and sub_domains[0] == 'www')
        is_www_weird = int(num_sub_domains > 0 and
                           bool(re.match(r'www.+', sub_domains[0])))
        path_len = len(path)
        domain_end_verdict = -1 * (domain_ending in UNTRUSTWORTHY_TLDS) + \
            1 * (domain_ending in TRUSTWORTHY_TLDS)

        feat_vec = np.array([
            is_https, num_main_domain_words,
            is_www, is_www_weird, path_len,
            domain_end_verdict
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
            url_data = url_tokenizer(url)
            feat_vec = self.__create_hand_picked_features__(url_data)
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
