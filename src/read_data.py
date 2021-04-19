from typing import Tuple, Dict
from os import listdir
from os.path import isdir, join, dirname, abspath
import pandas as pd
import numpy as np

CUR_DIR = dirname(abspath(__file__))
DATA_DIR = join(CUR_DIR, 'data')
DMOZ_DIR = join(DATA_DIR, 'dmoz')
WEBKB_DIR = join(DATA_DIR, 'webkb')
PHISHING_DIR = join(DATA_DIR, 'phishing')
TOKEN_EXPANSION_DIR = join(DATA_DIR, 'token_expansion')

DMOZ_BASE_FILENAME = 'URL Classification'
PHISHING_BASE_FILENAME = 'phishing_dataset'
BENIGN_BASE_FILENAME = 'benign_dataset'
PHISHING_EXTRA_FILENAME = 'phishing_extra'
TOKEN_EXPANSION_FILENAME = 'AcronymsFile.csv'


def is_valid(url_to_test: str) -> bool:
    '''Checks whether a given url seems to be a valid url'''
    return type(url_to_test) is str and url_to_test.startswith('http')


def filter_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Checks for all rows whether they contain an invalid url and filters away
    those that are invalid
    '''
    valid_indices = df['url'].apply(is_valid)
    return df[valid_indices]


def read_dmoz(use_sample=False) -> pd.DataFrame:
    '''Reads the DMOZ dataset and returns it as a DataFrame'''
    sample = '_sample' if use_sample else ''
    filename = f'{DMOZ_BASE_FILENAME}{sample}.csv'
    filedir = join(DMOZ_DIR, filename)
    df = pd.read_csv(filedir, names=['idx', 'url', 'label'])
    return filter_invalid_rows(df)


def read_phishing(use_sample=False) -> pd.DataFrame:
    '''Reads the phishing dataset and returns it as a DataFrame'''
    sample = '_sample' if use_sample else ''

    p_filename = f'{PHISHING_BASE_FILENAME}{sample}.csv'
    p_filedir = join(PHISHING_DIR, p_filename)
    phishing_df = pd.read_csv(p_filedir, names=['url'])
    phishing_df['label'] = 'phishing'

    b_filename = f'{BENIGN_BASE_FILENAME}{sample}.csv'
    b_filedir = join(PHISHING_DIR, b_filename)
    benign_df = pd.read_csv(b_filedir, names=['url'])
    benign_df['label'] = 'benign'

    df = pd.concat([phishing_df, benign_df])
    df.insert(0, 'idx', np.arange(len(df)))

    return filter_invalid_rows(df)


def read_phishing_extra(use_sample=False) -> pd.DataFrame:
    '''Reads the extra phishing dataset and returns it as a DataFrame'''
    sample = '_sample' if use_sample else ''
    filename = f'{PHISHING_EXTRA_FILENAME}{sample}.csv'
    filedir = join(PHISHING_DIR, filename)
    df = pd.read_csv(filedir)[['url']]
    df['label'] = 'phishing'
    df.insert(0, 'idx', np.arange(len(df)))
    return filter_invalid_rows(df)


def read_ilp(use_sample=False) -> pd.DataFrame:
    '''Reads the ILP 98 WebKB dataset and returns it as a DataFrame'''
    data, idx = [], 0
    sample = '_sample' if use_sample else ''
    webkb_dir = WEBKB_DIR + sample
    label_dirs = [d for d in listdir(webkb_dir) if isdir(join(webkb_dir, d))]
    for label in label_dirs:
        label_dir = join(webkb_dir, label)
        uni_dirs = [d for d in listdir(label_dir) if isdir(join(label_dir, d))]
        for uni in uni_dirs:
            label_uni_dir = join(label_dir, uni)
            urls = [url for url in listdir(label_uni_dir)
                    if url.startswith('http')]
            for url in urls:
                replaced_url = url.replace('^', '/')
                replaced_url = replaced_url.replace('http_', 'http:')
                replaced_url = replaced_url.replace('https_', 'https:')
                data.append([idx, replaced_url, label, uni])
                idx += 1
    df = pd.DataFrame(data, columns=['idx', 'url', 'label', 'uni'])
    return filter_invalid_rows(df)


def read_all_datasets(use_sample=False) -> Tuple[pd.DataFrame]:
    '''
    Reads the datasets and returns a 3-tuple of datasets (DMOZ, Phishing, ILP)
    in the form of DataFrames
    '''
    return (
        read_dmoz(use_sample=use_sample),
        read_phishing(use_sample=use_sample),
        read_ilp(use_sample=use_sample)
    )


def read_concat_datasets(use_sample=False) -> pd.DataFrame:
    '''
    Reads all the datasets and returns a single DataFrame consiting of all of
    them
    '''
    dmoz, phishing, ilp = read_all_datasets(use_sample=use_sample)
    ilp = ilp[['idx', 'url', 'label']]

    dmoz['dataset'] = 'dmoz'
    phishing['dataset'] = 'phishing'
    ilp['dataset'] = 'ilp'
    concatted = pd.concat([dmoz, phishing, ilp])
    return concatted


def read_token_expansion_dataset() -> Dict[str, str]:
    '''
    Generates a Dict[str, str] for token expansion where the key is the
    abbreviated token and the value is the phrase of the token.

    Returns:
        acronyms (Dict[str, str]): Dictionary of abbreviated tokens and their
                                   corresponding phrases.
    '''
    filepath = join(TOKEN_EXPANSION_DIR, TOKEN_EXPANSION_FILENAME)
    acronym_df = pd.read_csv(filepath, header=None, index_col=0,
                             dtype=str, na_filter=False)
    acronyms = {acronym: acronym_df.loc[acronym].values[0]
                for acronym in acronym_df.index.values}
    return acronyms
