import numpy as np
from pathlib import Path
from pos_tagging.utils import *


def main():

    # Raw data with ekphrasis
    raw_data_file = Path("input/raw_data.npy")

    if raw_data_file.is_file():
        raw_data = np.load(raw_data_file)
    else:
        raw_data = get_raw_data()
        np.save("input/raw_data", raw_data)   # for future use



    # POS data with NLTK
    pos_data_file = Path("input/pos_data.npy")

    if pos_data_file.is_file():
        pos_data = np.load(pos_data_file)
    else:
        pos_data = get_pos(raw_data)
        np.save("input/pos_data", pos_data)   # for future use



    # Generate embedding matrix for pos_data
    pos_emb_matrix = embedding_bin()



    # Split data to train and test
    pos_train, pos_test = split_train_test(pos_data)


    # Create output files
    np.save("outputs/pos_TRAIN", pos_train)
    np.save("outputs/pos_TEST", pos_test)
    np.save("outputs/pos_emb_matrix", pos_emb_matrix)

if __name__ == "__main__":
    print("Creating POS files")
    main()
