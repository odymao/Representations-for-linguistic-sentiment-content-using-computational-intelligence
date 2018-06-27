import fastText as ft
import numpy as np

word_index = np.load("word_index.npy")
word_index = word_index.item()  # to get rid of ndarray

model = ft.load_model('/my_data/wiki.en.bin')

max_nb_words = 50000
emb_dim = 300

num_words = min(max_nb_words, len(word_index))
embedding_matrix = np.zeros((len(word_index) + 1, emb_dim))

for word, i in word_index.items():
    if i >= max_nb_words:
        continue
    try:
        embedding_matrix[i] = model.get_word_vector(word)
    except:
        print("error in word " + word)

np.save("/output/embeddingMatrix_fastTextTRAIN_n_TEST_DATA300ALL-50-50000", embedding_matrix)
