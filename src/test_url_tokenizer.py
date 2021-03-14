import pytest
from url_tokenizer import url_raw_splitter, url_domains_handler, \
                          url_path_handler, url_args_handler


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
