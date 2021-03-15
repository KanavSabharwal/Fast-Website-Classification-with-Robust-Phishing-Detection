from typing import List, Any

import wordninja
import itertools

MIN_SPLIT_LEN = 4


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


def word_splitter(text: str) -> List[str]:
    '''
    Splits a string into multiple words mainly using wordninja, but keeps
    the string as it is if is shorter than or equal to the MIN_SPLIT_LEN

    Args:
        text (str): The string of text to split

    Returns:
        lst (List[str]): List of tokenized words
    '''
    if text:
        return [text] if len(text) <= MIN_SPLIT_LEN else wordninja.split(text)
    else:
        return []
