# extract stanford_TRAIN.npy, stanford_TEST.npy
# from raw_data

from pathlib import Path
from coreNLP.utils import *


def main(pad_length):
    # pad_length: how many sentences from the original tweet to keep in the padded data

    # Raw data with ekphrasis
    raw_data_file = Path("input/raw_data.npy")

    if raw_data_file.is_file():
        raw_data = np.load(raw_data_file)
    else:
        print("Error in getting raw data")

    # Getting sentiment from coreNLP
    corenlp_file = Path("input/coreNLP_data.npy")

    if corenlp_file.is_file():
        corenlp = np.load(corenlp_file)
    else:
        corenlp = get_corenlp(raw_data)
        np.save("input/coreNLP_data", corenlp)   # for future use

    # Get sentiments in each tweet
    data = get_sents(corenlp)


    # Pad sentiment data (max * 5)
    padded_data = pad_tweets(data, pad_length)  # max number of sentences in a tweet in this dataset: 12

    # Split sentiment data
    data_train, data_test = split_train_test(padded_data)


    # # Create output files
    np.save("outputs/stanford_TRAIN", data_train)
    np.save("outputs/stanford_TEST", data_test)


if __name__ == "__main__":
    MAX_SENTENCES = 12  # How many sentences from the original tweet to keep
    print("Using Standford's coreNLP to get sentiment distribution")
    main(MAX_SENTENCES)
