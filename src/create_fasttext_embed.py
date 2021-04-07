import pickle

from gensim.models.fasttext import FastText

from read_data import read_all_datasets
from self_trained_embeddings import sentence_handler_func

DMOZ, PHISHING, ILP = 'dmoz', 'phishing', 'ilp'

MIN_N = 2
MAX_N = 5
EPOCHS = 10
VECTOR_SIZE = 100

dmoz, phishing, ilp = read_all_datasets(use_sample=False)

DATASETS = {
    DMOZ: dmoz,
    PHISHING: phishing,
    ILP: ilp
}

# Code based on
# # https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html


def create_corpus_file(dataset_str: str) -> str:
    '''
    Takes a string representing the corpus to use, tokenizes it and writes a
    new file with the tokenized URLs each on a separate line. Returns the path
    to this file.
    '''
    corpus_file = f'sentences-{dataset_str}.txt'
    dataset = DATASETS[dataset_str]
    sentences = sentence_handler_func(dataset)
    with open(corpus_file, 'w') as f:
        f.writelines([' '.join(sentence) + '\n' for sentence in sentences])
    return corpus_file


def create_model_from_corpus(corpus_file: str) -> FastText:
    '''Reads the corpus file and trains a FastText model on it'''
    model = FastText(vector_size=VECTOR_SIZE)
    model.build_vocab(corpus_file=corpus_file)
    model.train(
        corpus_file=corpus_file, epochs=EPOCHS,
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words,
        min_n=MIN_N, max_n=MAX_N
    )
    return model


def main():
    for dataset_str in DATASETS.keys():
        print(dataset_str)
        corpus_file = create_corpus_file(dataset_str)
        model = create_model_from_corpus(corpus_file)
        wv_lookup_dataset = model.wv
        with open(f'embed-{dataset_str}.pickle', 'wb') as f:
            pickle.dump(wv_lookup_dataset, f)


if __name__ == '__main__':
    main()
