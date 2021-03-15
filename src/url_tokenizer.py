import re
from typing import List, Tuple

from util import flatten, word_splitter

# TODO: We should also handle HTML Encodings like %20 etc.
# We may use something like html.unescape('&pound;682m') for this
# import html

DomainData = Tuple[List[str], List[str], str]
ParamValPair = Tuple[str, str]
UrlData = Tuple[str, DomainData, List[str], List[ParamValPair]]


def url_tokenizer(url: str) -> UrlData:
    '''
    Takes a url as a string and returns a 4-tuple of the processed protocol,
    domains, path and arguments.

    Args:
        url (str): Full url of webpage

    Returns:
        protocol (str): Protocol, http or https
        domains (DomainData): A tuple consisting of a list of the sub-domains,
                              list of the main domain tokenized and the domain
                              ending
        path (List[str]): A list of the tokens in the path
        args (List[ParamValPair]): A list of the corresponding parameters
                                   and values in the URL
    '''
    protocol, domains_raw, path_raw, args_raw = url_raw_splitter(url)
    domains = url_domains_handler(domains_raw)
    path = url_path_handler(path_raw)
    args = url_args_handler(args_raw)
    return (protocol, domains, path, args)


def url_raw_splitter(url: str) -> Tuple[str]:
    '''
    Takes a url as a string and returns a 4-tuple of the raw protocol,
    domains, path and arguments.

    Args:
        url (str): Full url of webpage

    Returns:
        raw_values (Tuple[str]): 4-tuple of the raw protocol, domains,
                                 path and arguments

    Examples:
        >>> url_raw_splitter('http://www.sub.web.com/path1/path2?arg=val')
        ('http', 'www.sub.web.com', '/path1/path2', 'arg=val')
    '''
    # RegEx partly based on https://stackoverflow.com/a/3809435/9248793
    url_regex = re.compile(r'''
        (https?):\/\/                                   # http s
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


def url_domains_handler(url_domains: str) -> DomainData:
    '''
    Splits the domain part of the URL to individual tokens

    Args:
        url_domains (str): Domains part of url of webpage

    Returns:
        sub_domains (List[str]): List of subdomains, i.e. ['www', 'blog']
        main_domain (List[str]): Main domain, i.e. ['geo', 'cities']
        domain_ending (str): The domain ending, i.e. 'com' or 'net'

    Examples:
        >>> url_domains_handler('geocities.com')
        ([], ['geo', 'cities'], 'com')
        >>> url_domains_handler('www.members.tripod.net')
        (['www', 'members'], ['tripod'], 'net')
    '''
    splitted = url_domains.split('.')
    sub_domains = flatten([word_splitter(w) for w in splitted[:-2]])
    main_domain = word_splitter(splitted[-2])
    domain_ending = splitted[-1]
    return (sub_domains, main_domain, domain_ending)


def url_path_handler(url_path: str) -> List[str]:
    '''
    Splits the path part of the url

    Args:
        url_path (str): Path part of url of webpage

    Returns:
        paths (List[str]): List of tokenized paths

    Examples:
        >>> url_path_handler('/path1/path2/page.html')
        ['path', '1', 'path', '2', 'page', 'html']
        >>> url_path_handler('/')
        []
    '''
    return flatten([word_splitter(token) for token in url_path.split('/')
                    if token])


def url_args_handler(url_args: str) -> List[ParamValPair]:
    '''
    Tokenizes the parameter-value pairs in the parameter part of the url

    Args:
        url_args (str): Parameter part of url of webpage

    Returns:
        paths (ParamValPair): List of param-val pairs as 2tuple of lists. If no
                              val is given, the second value in the tuple is
                              the empty list []

    Examples:
        >>> url_args_handler('sid=4')
        [('sid', '4')]
        >>> url_args_handler('sid=4&amp;ring=hent&amp;list')
        [(['sid'], ['4']), (['ring'], ['hent']), (['list'], [])]
        >>> url_args_handler('')
        []
    '''
    if len(url_args) == 0:
        return []

    pair_list = []
    for pair in re.split(r'(?:&amp;)|;|&', url_args):
        splitted = pair.split('=')[:2]
        param, val = (splitted[0], '') if len(splitted) == 1 else splitted
        param_val_tup = (word_splitter(param), word_splitter(val))
        pair_list.append(param_val_tup)
    return pair_list
