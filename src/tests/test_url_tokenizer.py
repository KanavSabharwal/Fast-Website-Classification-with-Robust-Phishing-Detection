from url_tokenizer import url_raw_splitter, url_domains_handler, \
                          url_path_handler, url_args_handler, \
                          url_html_decoder, flatten_url_data, \
                          url_tokenizer, expand_token, expand_url_tokens

from read_data import read_token_expansion_dataset

class TestUrlRawSplitter:
    def test_basic_url(self):
        vals = url_raw_splitter('http://www.some-basic_8site.com')
        assert len(vals) == 4
        protocol, domains, path, args = vals
        assert protocol == 'http'
        assert domains == 'www.some-basic_8site.com'
        assert path == ''
        assert args == ''

    def test_path_url(self):
        vals = url_raw_splitter('http://www.some-basic_8site.com/some/path/')
        assert len(vals) == 4
        protocol, domains, path, args = vals
        assert protocol == 'http'
        assert domains == 'www.some-basic_8site.com'
        assert path == '/some/path/'
        assert args == ''

    def test_args_url(self):
        vals = url_raw_splitter('http://some-site.com/some/path?arg=val')
        assert len(vals) == 4
        protocol, domains, path, args = vals
        assert protocol == 'http'
        assert domains == 'some-site.com'
        assert path == '/some/path'
        assert args == 'arg=val'


class TestFlattenUrlData:
    def test_basic_url(self):
        url_data = url_tokenizer('http://test.com/')
        exp_lst = ['http', 'test', 'com']
        assert flatten_url_data(url_data) == exp_lst

    def test_comprehensive_url(self):
        url = 'http://some.test.com/path.html?arg1=val1'
        url_data = url_tokenizer(url)
        exp_lst = ['http', 'some', 'test', 'com', 'path',
                   'html', 'arg1', 'val1']
        assert flatten_url_data(url_data) == exp_lst


class TestUrlDomainsHandler:
    def test_no_subdomain(self):
        vals = url_domains_handler('site.com')
        assert len(vals) == 3
        sub_domains, main_domain, domain_ending = vals
        assert sub_domains == []
        assert main_domain == ['site']
        assert domain_ending == 'com'

    def test_basic_domain(self):
        vals = url_domains_handler('www.site.com')
        assert len(vals) == 3
        sub_domains, main_domain, domain_ending = vals
        assert sub_domains == ['www']
        assert main_domain == ['site']
        assert domain_ending == 'com'

    def test_many_subdomains(self):
        vals = url_domains_handler('rpo.library.part8.site.com')
        assert len(vals) == 3
        sub_domains, main_domain, domain_ending = vals
        assert sub_domains == ['rpo', 'library', 'part', '8']
        assert main_domain == ['site']
        assert domain_ending == 'com'

    def test_many_subdomains_splitted_main(self):
        vals = url_domains_handler('rpo.library.part8.geocities.com')
        assert len(vals) == 3
        sub_domains, main_domain, domain_ending = vals
        assert sub_domains == ['rpo', 'library', 'part', '8']
        assert main_domain == ['geo', 'cities']
        assert domain_ending == 'com'


class TestUrlPathHandler:
    def test_empty(self):
        assert url_path_handler('') == []

    def test_single_slash(self):
        assert url_path_handler('/') == []

    def test_leading_slash(self):
        assert url_path_handler('/test/') == ['test']

    def test_basic_path(self):
        assert url_path_handler('/some/long/path/with/weird/2783/23') == \
            ['some', 'long', 'path', 'with', 'weird', '2783', '23']

    def test_path_multiword(self):
        assert url_path_handler('/some/mediumlengthpath/') == \
            ['some', 'medium', 'length', 'path']
    
    def test_aite(self):
        assert url_path_handler('/mbraun@ameritech.net/') == \
            ['m', 'braun', 'a', 'merit', 'ech', 'net', '@']


class TestUrlArgsHandler:
    def test_empty(self):
        assert url_args_handler('') == []

    def test_single(self):
        args = url_args_handler('sid=4')
        assert args == [(['sid'], ['4'])]

    def test_multiple(self):
        args = url_args_handler('sid=4&amp;ring=hent&amp;id=2')
        assert args == [(['sid'], ['4']), (['ring'], ['hent']),
                        (['id'], ['2'])]

    def test_multiple_with_empty(self):
        args = url_args_handler('sid=4&amp;ring=hent&amp;id=2&amp;list')
        assert args == [(['sid'], ['4']), (['ring'], ['hent']),
                        (['id'], ['2']), (['list'], [])]

    def test_multiword(self):
        args = url_args_handler('amultiwordparam=multiwordvalue')
        assert args == [(['a', 'multi', 'word', 'param'],
                         ['multi', 'word', 'value'])]


class TestUrlHtmlEncoder:
    def test_simple_website(self):
        encoded_url = 'http://e.webring.com/hub?sid=&amp;ring=hentff98&amp;id=&amp'
        decoded_url = 'http://e.webring.com/hub?sid=&ring=hentff98&id=&'
        assert url_html_decoder(encoded_url) == decoded_url

    def test_multiple_encodings(self):
        encoded_url = 'http://www.asstr.org/janice%20and%20kirk%27s'
        decoded_url = "http://www.asstr.org/janice and kirk's"
        assert url_html_decoder(encoded_url) == decoded_url


class TestTokenExpansion:
    def test_expanded_token(self):
        Acrony_dict = dict()
        Acrony_dict = read_token_expansion_dataset()

        args = expand_token(Acrony_dict,'cs')
        assert args == 'computer science'

        args = expand_token(Acrony_dict,'nlp')
        assert args == 'natural language processing'

        args = expand_token(Acrony_dict,'nos')
        assert args == 'network operating system'


    def test_url_tuple_expansion(self):
        Acrony_dict = dict()
        Acrony_dict = read_token_expansion_dataset()

        args = expand_url_tokens(Acrony_dict,('http', (['www'], ['a', 'odon', 'line'], 'org'), ['chsl', 'cs', 'sl', 'htm'], []))
        assert args == ('http', (['world', 'wide', 'web'], ['a', 'odon', 'line'], 'org'), ['chsl', 'computer', 'science', 'sierra', 'leone', 'htm'], [])

        args = expand_url_tokens(Acrony_dict,('http', (['ed', 'web', '3', 'educ'], ['msu'], 'edu'), ['ysi'], [('sid','nlp')]))
        assert args == ('http', (['ed', 'web', '3', 'educ'], ['michigan', 'state', 'university'], 'edu'), ['young', 'scots', 'for', 'independence'], [(['sid'], ['natural', 'language', 'processing'])])



