import re
import html
from typing import List, Tuple


DomainData = Tuple[List[str], str, str]
ParamValPair = Tuple[str, str]
UrlData = Tuple[str, DomainData, List[str], List[ParamValPair]]


# TODO: We should also handle HTML Encodings like %20 etc.
# We may use something like html.unescape('&pound;682m') for this


def url_tokenizer(url: str) -> UrlData:
    '''
    Takes a url as a string and returns a 4-tuple of the processed protocol,
    domains, path and arguments.

    Args:
        url (str): Full url of webpage

    Returns:
        protocol (str): Protocol, http or https
        domains (DomainData): A tuple consisting of a list of the sub-domains,
                              the main domain and the domain ending
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
    match = url_regex.match(url)
    assert match, f'Error matching url: {url}'
    raw_values = match.groups()
    return raw_values


def url_domains_handler(url_domains: str) -> DomainData:
    '''
    Splits the domain part of the URL to individual tokens and returns the

    Args:
        url_domains (str): Domains part of url of webpage

    Returns:
        sub_domains (List[str]): List of subdomains, i.e. ['www', 'blog']
        main_domain (str): Main domain, i.e. 'google'
        domain_ending (str): The domain ending, i.e. 'com' or 'net'

    Examples:
        >>> url_domains_handler('google.com')
        ([], 'google', 'com')
        >>> url_domains_handler('www.members.tripod.net')
        (['www', 'members'], 'tripod', 'net')
    '''
    splitted = url_domains.split('.')
    sub_domains = splitted[:-2]
    main_domain = splitted[-2]
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
        ['path1', 'path2', 'page.html']
        >>> url_path_handler('/')
        []
    '''
    paths = [token for token in url_path.split('/') if token]
    return paths


def url_args_handler(url_args: str) -> List[ParamValPair]:
    '''
    Tokenizes the parameter-value pairs in the parameter part of the url

    Args:
        url_args (str): Parameter part of url of webpage

    Returns:
        paths (ParamValPair): List of param-val pairs as tuple. If no val is
                              given, the second value in the tuple is None

    Examples:
        >>> url_args_handler('sid=4')
        [('sid', '4')]
        >>> url_args_handler('sid=4&amp;ring=hent&amp;id=2&amp;list')
        [('sid', '4'), ('ring', 'hent'), ('id', '2'), ('list', None)]
        >>> url_args_handler('')
        []
    '''
    if len(url_args) == 0:
        return []

    pair_list = []
    for pair in url_args.split('&amp;'):
        splitted = pair.split('=')
        param_arg = (splitted[0], None) if len(splitted) == 1 else tuple(splitted)
        pair_list.append(param_arg)
    return pair_list


def word_expander(text: str) -> List[str]:
    '''
    Takes a string of text that may contain concatenated words and tries to
    expand it to a list of separate tokens.

    Args:
        text (str): Text to process

    Returns:
        tokens (List[str]): List of tokenized words

    Examples:
        >>> word_expander('animaladventures')
        ['animal', 'adventures']
        >>> word_expander('women')
        ['women']
    '''
    # TODO: Implement this
    if len(text) == 0:
        return []

    tokens = [text]
    return tokens
