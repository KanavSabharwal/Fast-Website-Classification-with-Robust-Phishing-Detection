from os import listdir
from os.path import isdir, join
import pandas as pd


DATA_DIR = 'data'
DMOZ_DIR = join(DATA_DIR, 'dmoz')
WEBKB_DIR = join(DATA_DIR, 'webkb')
DMOZ_BASE_FILENAME = 'URL Classification'


def read_dmoz(use_sample=False) -> pd.DataFrame:
    '''Reads the DMOZ dataset and returns it as a DataFrame'''
    sample = '_sample' if use_sample else ''
    filename = f'{DMOZ_BASE_FILENAME}{sample}.csv'
    filedir = join(DMOZ_DIR, filename)
    df = pd.read_csv(filedir, names=['idx', 'url', 'label'])
    return df



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
                data.append([idx, replaced_url, label, uni])
                idx += 1
    return pd.DataFrame(data, columns=['idx', 'url', 'label', 'uni'])


def read_phishing(use_sample=False) -> pd.DataFrame:
    '''Reads the phishing dataset and returns it as a DataFrame'''
    # TODO: Implement this
    pass
