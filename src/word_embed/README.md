# Word Embeddings

In this folder, you should place the word embeddings. They are not included in the git repository due to their large size. At the moment, this project supports two different types.

* [ConceptNet]: Download `numberbatch-19.08.txt.gz`. We use multilingual.
* [GloVe]: Download the `glove.42B.300d.zip`-file.

You should download these, and place them in the folder such that the directory structure from this folder is:

```bash
word_embed/
	README.md  # this file
	conceptnet/
		numberbatch-19.08.txt
	glove/
		glove.42B.300d.txt
	sample/
		sample.txt  # used for testing
```



[GloVe]: https://nlp.stanford.edu/projects/glove/
[ConceptNet]: https://github.com/commonsense/conceptnet-numberbatch#Downloads