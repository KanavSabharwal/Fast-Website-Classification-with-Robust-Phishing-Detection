import re
from typing import List, Tuple

from util import flatten, flatten_twice, word_splitter
import html
import urllib.parse

from read_data import read_token_expansion_dataset


DomainData = Tuple[List[str], List[str], str]
ParamValPair = Tuple[str, str]
UrlData = Tuple[str, DomainData, List[str], List[ParamValPair]]

Acrony_dict = dict()  # the dictionary contains abbreviated tokens and their corresponding phrases. 
Acrony_dict = read_token_expansion_dataset()

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
    url_decoded = url_html_decoder(url)
    protocol, domains_raw, path_raw, args_raw = url_raw_splitter(url_decoded)
    domains = url_domains_handler(domains_raw)
    path = url_path_handler(path_raw)
    args = url_args_handler(args_raw)

    
    # expand tokens in the url tuple.Remove the comment symbol '#' if we need to expand the token.
    #url_tuple = (protocol, domains, path, args)
    #url_tuple_expanded = expand_url_tokens(Acrony_dict,url_tuple)
    #protocol, domains, path, args = url_tuple_expanded
    


    return (protocol, domains, path, args)


def flatten_url_data(url_data: UrlData) -> List[str]:
    '''
    Helper function to transform the 4-tuple of UrlData returned by
    url_tokenizer into a simple list of strings. Can be helpful to simplify
    the problem if the position of words is not relevant.

    Args:
        url_data (UrlData): The UrlData 4-tuple returned by url_tokenizer

    Returns:
        words (List[str]): A flat list of all the words

    Examples:
        >>> url_data = url_tokenizer('http://some.test.com/path')
        >>> flatten_url_data(url_data)
        ['http', 'some', 'test', 'com', 'path']
    '''
    protocol, domains, path, args = url_data
    sub_domain, main_domain, tld = domains
    words = ([protocol] + sub_domain + main_domain + [tld]
             + path + flatten_twice(args))
    return words


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
    token_lst = flatten([word_splitter(token) for token in url_path.split('/')
                        if token])
    if url_path.find('@') >= 0:
        token_lst.append('@')
    return token_lst


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
        [(['sid'], ['4'])]
        >>> url_args_handler('sid=4&amp;ring=hent&amp;list')
        [(['sid'], ['4']), (['ring'], ['hent']), (['list'], [])]
        >>> url_args_handler('')
        []
    '''
    if len(url_args) == 0:
        return []

    pair_list = []
    for pair in re.split(r'(?:&amp;)|;|&|\\', url_args):
        splitted = pair.split('=')[:2]
        param, val = (splitted[0], '') if len(splitted) == 1 else splitted
        param_val_tup = (word_splitter(param), word_splitter(val))
        pair_list.append(param_val_tup)
    return pair_list


def url_html_decoder(raw_url: str) -> str:
    '''
    Replaces all HTML encodings with their corresponding character to give the
    decoded URL string

    Args:
        raw_url (str): The original full url

    Returns:
        decoded_url (str): The decoded url

    Examples:
        >>> url_html_decoder('http://e.webring.com/hub?sid=&amp;ring=hentff98&amp;id=&amp')
        'http://e.webring.com/hub?sid=&ring=hentff98&id=&'
        >>> url_html_decoder('http://www.asstr.org/janice%20and%20kirk%27s')
        "http://www.asstr.org/janice and kirk's"
    '''
    unquoted_url = urllib.parse.unquote(raw_url)
    decoded_url = html.unescape(unquoted_url)
    return decoded_url



def expand_token(Acrony_dict,token) -> str:
    '''
    get token's corresponding phrase.

    args:
        Acrony_dict: the dictionary contains abbreviated tokens and their corresponding phrases. 
                    It's generated from function "generate_Acrony_dict".
        token: the token you'd like to expand.
    returns:
        the expanded token.        
    '''

    token = token.lower()
    token_expansion = None
    if token in Acrony_dict.keys():
        token_expansion = Acrony_dict[token]
    else:
        token_expansion = token

    return token_expansion



def expand_url_tokens(Acrony_dict,url_tuple):
    '''
    expand the tokens in the url tuple
    args:
        Acrony_dict: the dictionary contains abbreviated tokens and their corresponding phrases. 
                    It's generated from function "generate_Acrony_dict".
    url_tuple:(protocol, domains, path, args)
        protocol (str): Protocol, http or https
        domains (DomainData): A tuple consisting of a list of the sub-domains,
                              list of the main domain tokenized and the domain
                              ending
        path (List[str]): A list of the tokens in the path
        args (List[ParamValPair]): A list of the corresponding parameters
                                   and values in the URL
    returns:
        url tuple with expanded tokens
    '''
    url_list = list(url_tuple)
    protocol, domains, path, args = url_list

    domain_list = list(domains)
    sub_domain, main_domain, domain_ending = domain_list

    sub_domain = [expand_token(Acrony_dict,token).split(' ') for token in sub_domain]
    main_domain = [expand_token(Acrony_dict,token).split(' ') for token in main_domain]
    path = [expand_token(Acrony_dict,token).split(' ') for token in path]
    
    for i in range(len(args)):
        args[i] = tuple([expand_token(Acrony_dict,token).split(' ') for token in args[i]])


    #Removing nested lists
    sub_domain = flat(sub_domain)
    main_domain = flat(main_domain)
    path = flat(path)
    args = flat(args)

    url_tuple_new = (protocol,(sub_domain,main_domain,domain_ending),path,args)
    
    return url_tuple_new


def flat(a):
    '''
    function for removing nested lists. This function is for handling situation like [['computer','scicence']] after expanding tokens.
    args: a list like [[a],[b]]
    return: the nested list [a,b]
    '''
    l = []
    for i in a:
        if type(i) is list:
            for j in i:
                l.append(j)
        else:
            l.append(i)
    return l
