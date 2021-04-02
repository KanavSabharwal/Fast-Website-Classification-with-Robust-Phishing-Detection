import pickle

from read_data import read_all_datasets, read_phishing_extra


def pickle_main_datasets():
    '''
    Reads the main three datasets and pickles a dictionary of shuffled
    DataFrames of them
    '''
    dmoz, phishing, ilp = read_all_datasets(use_sample=False)
    data = {
        'dmoz': dmoz.sample(frac=1),
        'phishing': phishing.sample(frac=1),
        'ilp': ilp.sample(frac=1)
    }
    with open('datasets.pkl', 'wb') as f:
        pickle.dump(data, f)


def pickle_extra_phishing():
    '''
    Reads the extra phishing datasets and pickles a shuffled DataFrame of it
    '''
    phishing_extra = read_phishing_extra().sample(frac=1)
    with open('phishing_extra.pkl', 'wb') as f:
        pickle.dump(phishing_extra, f)


def main():
    pickle_main_datasets()
    pickle_extra_phishing()


if __name__ == '__main__':
    main()
