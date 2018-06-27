# Download glove.6B.300d.txt and place it in the "glove" dir from http://nlp.stanford.edu/data/glove.6B.zip
# Download word2vec/GoogleNews-vectors-negative300.bin and place it in the "word2vec" dir from https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download

# extract data_TRAIN.npy,
#         data_TEST.npy,
#         embeddingMatrix_gloveTRAIN_n_TEST_DATA300ALL-50-50000.npy,
#         embeddingMatrix_word2vecTRAIN_n_TEST_DATA300ALL-50-50000.npy,
#         labels_TRAIN.npy,
#         labels_TEST.npy,
# from
# input/final_tweets_no_duplicates_no_NAs_plus_TEST.txt

import numpy as np
from pathlib import Path
from data.utils import *


def main(w2v_limited, pad):

    # Raw data with ekphrasis
    raw_data_file = Path("input/raw_data.npy")

    if raw_data_file.is_file():
        raw_data = np.load(raw_data_file)
    else:
        raw_data = get_raw_data()
        np.save("input/raw_data", raw_data)   # for future use


    # Lexicons data
    lexicons_data = get_lexicons_data()
    np.save("input/lexicons_data", lexicons_data)  # for future use

    # Extract labels
    labels_train = get_labels()
    labels_test = get_true_labels()

    # Extract data
    data, word_index = get_data(raw_data, pad, max_nb_words=50000, max_seq_length=50)
    np.save("data/floyd_aux/word_index", word_index)  # saving word_index for extracting fastText embedding mat via floydhub (see floyd_aux/README.txt)

    # Create embedding matrixes
    emb_matrix_glove = create_embedding_matrix_glove(word_index, max_nb_words=50000, emb_dim=300)
    emb_matrix_w2v = create_embedding_matrix_w2v(word_index, w2v_limited, max_nb_words=50000, emb_dim=300)

    # Split data
    data_train, data_test = split_train_test(data, pad)

    # # Create output files
    np.save("outputs/data_TRAIN", data_train)
    np.save("outputs/data_TEST", data_test)
    np.save("outputs/embeddingMatrix_gloveTRAIN_n_TEST_DATA300ALL-50-50000", emb_matrix_glove)
    np.save("outputs/embeddingMatrix_word2vecTRAIN_n_TEST_DATA300ALL-50-50000", emb_matrix_w2v)
    np.save("outputs/labels_TRAIN", labels_train)
    np.save("outputs/labels_TEST", labels_test)

if __name__ == "__main__":
    print("Creating data, labels and embedding matrix files")
    main()
