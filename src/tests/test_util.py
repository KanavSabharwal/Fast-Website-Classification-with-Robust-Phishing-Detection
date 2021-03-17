from util import flatten, flatten_twice, word_splitter

class TestFlatten:
    def test_flatten_lst(self):
        assert flatten([[1, 1], [1, 1]]) == [1, 1, 1, 1]

    def test_flatten_tuple(self):
        assert flatten([(1, 1), (1, 1)]) == [1, 1, 1, 1]


class TestFlattenTwice:
    def test_flatten_lst(self):
        assert flatten_twice([[[1], [1]], [[1], [1, 1]]]) == [1, 1, 1, 1, 1]

    def test_flatten_tuple(self):
        assert flatten_twice([[(1, 1)], [(1,), (1, 1)]]) == [1, 1, 1, 1, 1]


class TestWordSplitter:
    def test_empty_word(self):
        assert word_splitter('') == []

    def test_short_word(self):
        assert word_splitter('abc') == ['abc']

    def test_hyphenated_word(self):
        assert word_splitter('some-word') == ['some', 'word']

    def test_non_hyphenated_word(self):
        assert word_splitter('someword') == ['some', 'word']
