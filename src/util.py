from typing import List, Any

import wordninja
import itertools

MIN_SPLIT_LEN = 5


def flatten(lst_lst: List[List[Any]]) -> List[Any]:
    '''
    Takes a list of lists of any type and flattens it to a single list

    Args:
        lst_lst (List[List[Any]]): List of lists

    Returns:
        lst (List[Any]): Flattened (1D) list
    '''
    return list(itertools.chain(*lst_lst))


def flatten_twice(lst_lst_lst: List[List[List[Any]]]) -> List[Any]:
    '''
    Takes a list of lists of any type and flattens it to a single list

    Args:
        lst_lst (List[List[Any]]): List of lists

    Returns:
        lst (List[Any]): Flattened (1D) list
    '''
    return flatten(flatten(lst_lst_lst))


def word_splitter(text: str, min_split_len: int = MIN_SPLIT_LEN) -> List[str]:
    '''
    Splits a string into multiple words mainly using wordninja, but keeps
    the string as it is if is shorter than or equal to the min_split_len

    Args:
        text (str): The string of text to split
        min_split_len (int): The minimum word length before splitting

    Returns:
        lst (List[str]): List of tokenized words
    '''
    if text:
        return wordninja.split(text) if len(text) >= min_split_len else [text]
    else:
        return []
