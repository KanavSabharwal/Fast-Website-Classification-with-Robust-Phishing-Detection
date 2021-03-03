import re
from typing import List


def tokenize_url(url: str) -> List[str]:
    '''Takes a url as a string and returns a list of the tokenized parts'''
    # TODO: Refine tokenization based on examples of URLs
    url_regex = re.compile(r'''
        (https?)?   # http s
        (?:://)?    # optional :// part
        (www)?      # www
    ''', re.DOTALL | re.VERBOSE)
    match = url_regex.match(url)
    return match.groups()
