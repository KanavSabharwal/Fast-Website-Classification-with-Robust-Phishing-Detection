# Word Embeddings

Previously, this project used word embeddings stored locally here. For better compatibility with Colab we've changed the code to use `gensim` instead where we download the embeddings on demand. Currently, it is possible to initialize the `UrlFeaturizer` with Conceptnet Numberbatch, GloVe, FastText and Word2Vec embeddings.