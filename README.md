# NLP Project

Predicting webpages without the actual webpage.


## Getting started with the code

To install dependencies, run

```bash
$ cd src
$ pip install -r requirements.txt
```

Before you can actually run most of the code, you need to download and organize the appropriate datasets and word embeddings. See [`src/data/README.md`][data-readme] and [`src/word_embed/README.md`][word-readme] for how to do this.

## Testing

This project uses `pytest` for testing. After having installed dependencies, you can simply run

```bash
$ pytest
```

to run all tests.


[data-readme]: ./src/data/README.md
[word-readme]: ./src/word_embed/README.md