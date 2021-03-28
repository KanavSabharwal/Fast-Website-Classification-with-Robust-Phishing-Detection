import pytest
import numpy as np
from featurizer import UrlFeaturizer, SAMPLE


def create_feat(sub_domain_max_len, main_domain_max_len,
                path_max_len, arg_max_len):
    return UrlFeaturizer(SAMPLE, verbose=False,
                         sub_domain_max_len=sub_domain_max_len,
                         main_domain_max_len=main_domain_max_len,
                         path_max_len=path_max_len,
                         arg_max_len=arg_max_len)


class TestUrlRawSplitter:
    def test_errors_on_undefined_embeddings(self):
        with pytest.raises(ValueError):
            UrlFeaturizer('undefined-embed')

    def test_N_property(self):
        feat = create_feat(2, 3, 4, 5)
        assert feat.N == (2 + 3 + 4 + 5 + 1)

    def test_unknown_vec(self):
        fst, snd = create_feat(1, 1, 1, 1).unknown_vec
        assert fst == -np.mean(np.arange(1, 9))
        assert snd == np.mean(np.arange(1, 9))

    def test_ret_type_single(self):
        feat = create_feat(1, 1, 1, 1)
        ret = feat.featurize('http://test.com')
        assert len(ret) == 2
        vec, mat = ret
        assert len(vec.shape) == 1
        assert len(mat.shape) == 2

    def test_ret_type_multiple(self):
        feat = create_feat(1, 1, 1, 1)
        ret = feat.featurize(['http://test.com', 'http://test.com'])
        assert len(ret) == 2
        assert len(ret[0]) == 2
        vec, mat = ret[0]
        assert len(vec.shape) == 1
        assert len(mat.shape) == 2

    def test_feature_vec(self):
        feat = create_feat(1, 1, 1, 1)
        # is_https, num_main_domain_words, num_sub_domains is_www,
        # is_www_weird, path_len, domain_end_verdict
        # domains_num_len, path_num_len, args_num_len, total_num_len
        vec, _ = feat.featurize('http://test.com')
        assert np.allclose(vec, np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]))

        vec, _ = feat.featurize('https://test-a-domain.xyz')
        assert np.allclose(vec, np.array([1, 3, 0, 0, 0, 0, -1, 0, 0, 0, 0]))

        vec, _ = feat.featurize('https://www.sub1.sub2.test.unknown')
        assert np.allclose(vec, np.array([1, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0]))

        vec, _ = feat.featurize('https://wwws.test.com/some/long/long/path')
        assert np.allclose(vec, np.array([1, 1, 1, 0, 1, 4, 1, 0, 0, 0, 0]))

        vec, _ = feat.featurize('http://www2.117.ne.jp/~mb1996ax/enadc.html')
        assert np.allclose(vec, np.array([0, 1, 2, 0, 1, 7, 0, 3, 4, 0, 7]))

    def test_word_embed_matrix_size(self):
        feat = create_feat(1, 1, 1, 1)
        _, mat = feat.featurize('http://test.com')
        assert mat.shape == (5, 2)

        feat = create_feat(4, 10, 20, 30)
        _, mat = feat.featurize('http://test.com')
        assert mat.shape == (4 + 10 + 20 + 30 + 1, 2)

    def test_set_hyperparam(self):
        feat = create_feat(1, 1, 1, 1)
        feat.set_hyperparams(4, 10, 20, 30)
        _, mat = feat.featurize('http://test.com')
        assert mat.shape == (4 + 10 + 20 + 30 + 1, 2)

    def test_word_embed_matrix_values(self):
        feat = create_feat(1, 2, 1, 2)
        _, mat = feat.featurize('http://unk.test.com?arg=val')

        assert np.allclose(mat, np.array([
            feat.unknown_vec,  # unknown token, should be average vector
            feat.embeddings_index['test'],
            [-0.0, 0.0],  # non-present 2nd index main domain, should be zeros
            feat.embeddings_index['com'],
            [-0.0, 0.0],  # non-present path, should be zeros
            feat.embeddings_index['arg'],
            feat.embeddings_index['val'],
        ]))
