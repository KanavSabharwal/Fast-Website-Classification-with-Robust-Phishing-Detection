import pickle

from read_data import read_all_datasets


def main():
    # Read and shuffle data
    dmoz, phishing, ilp = read_all_datasets(use_sample=False)
    dmoz = dmoz.sample(frac=1)
    phishing = phishing.sample(frac=1)
    ilp = ilp.sample(frac=1)

    data = {
        'dmoz': dmoz,
        'phishing': phishing,
        'ilp': ilp
    }

    with open('datasets.pkl', 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main()
