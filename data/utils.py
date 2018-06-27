from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from gensim.models.keyedvectors import KeyedVectors
from pathlib import Path
from collections import Counter
from itertools import chain, repeat, islice
import numpy as np


def get_raw_data():
    nb_tweets = 51075  # number of tweets to read from .txt (max:51075)
    # nb_tweets = 100  # number of tweets to read from .txt (max:51075)
    # nb_tweets = 1  # number of tweets to read from .txt (max:51075)
    shift = 0  # starting point

    # read .txt and make a data array
    with open('input/final_tweets_no_duplicates_no_NAs_plus_TEST.txt', encoding="utf8") as f:
        data = [x.strip().split('\t') for x in f]  # data:489x3

    # build a tokenizer and a text processor
    def ws_tokenizer(text):
        return text.split()

    text_processor = TextPreProcessor(
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
        annotate={"hashtag", "allcaps", "elongated", "repeated", 'emphasis', 'censored'},
        fix_text=True,
        segmenter="twitter",
        corrector="twitter",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True, censored=False).tokenize,
        # tokenizer=ws_tokenizer,
        dicts=[emoticons]
    )

    # using ekphrasis api to alter the initial tweets
    print('starting...')
    # print(data[9582][2])
    for i in range(shift, shift + nb_tweets):
        data[i][2] = " ".join(text_processor.pre_process_doc(data[i][2]))

    for i in range(shift, shift + nb_tweets):
        data[i][2] = data[i][2].replace("<hashtag>", '')
        data[i][2] = data[i][2].replace("</hashtag>", '')
        data[i][2] = data[i][2].replace("<allcaps>", '')
        data[i][2] = data[i][2].replace("<elongated>", '')
        data[i][2] = data[i][2].replace("<repeated>", '')
        data[i][2] = data[i][2].replace("<emphasis>", '')
        data[i][2] = data[i][2].replace("<censored>", '')

        data[i][2] = data[i][2].replace("<kiss>", '')
        data[i][2] = data[i][2].replace("<happy>", '')
        data[i][2] = data[i][2].replace("<laugh>", '')
        data[i][2] = data[i][2].replace("<sad>", '')
        data[i][2] = data[i][2].replace("<surprise>", '')
        data[i][2] = data[i][2].replace("<wink>", '')
        data[i][2] = data[i][2].replace("<tong>", '')
        data[i][2] = data[i][2].replace("<annoyed>", '')
        data[i][2] = data[i][2].replace("<seallips>", '')
        data[i][2] = data[i][2].replace("<angel>", '')
        data[i][2] = data[i][2].replace("<devil>", '')
        data[i][2] = data[i][2].replace("<highfive>", '')
        data[i][2] = data[i][2].replace("<heart>", '')

        data[i][2] = data[i][2].replace("<url>", '')
        data[i][2] = data[i][2].replace("<email>", '')
        data[i][2] = data[i][2].replace("<percent>", '')
        data[i][2] = data[i][2].replace("<money>", '')
        data[i][2] = data[i][2].replace("<phone>", '')
        data[i][2] = data[i][2].replace("<user>", '')
        data[i][2] = data[i][2].replace("<time>", '')
        data[i][2] = data[i][2].replace("<date>", '')
        data[i][2] = data[i][2].replace("<number>", '')

        # beautify text
        data[i][2] = " ".join(text_processor.pre_process_doc(data[i][2]))  # ka9e fora megalwnei to censored
        data[i][2] = " ".join(text_processor.pre_process_doc(data[i][2]))
        data[i][2] = data[i][2].replace("<number>", '')
        data[i][2] = data[i][2].replace("<elongated>", '')
        data[i][2] = data[i][2].replace("<date>", '')
        data[i][2] = " ".join(text_processor.pre_process_doc(data[i][2]))

    raw_data = []

    for i in range(shift, shift + nb_tweets):
        raw_data.append(data[i][2])

    return raw_data


def get_lexicons_data():
    nb_tweets = 51075  # number of tweets to read from .txt (max:51075)
    # nb_tweets = 100  # number of tweets to read from .txt (max:51075)
    # nb_tweets = 1  # number of tweets to read from .txt (max:51075)
    shift = 0  # starting point

    # read .txt and make a data array
    with open('input/final_tweets_no_duplicates_no_NAs_plus_TEST.txt', encoding="utf8") as f:
        data = [x.strip().split('\t') for x in f]  # data:489x3

    # build a tokenizer and a text processor
    def ws_tokenizer(text):
        return text.split()

    text_processor = TextPreProcessor(
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
        annotate={"allcaps", "elongated", "repeated", 'emphasis', 'censored'},
        fix_text=True,
        segmenter="twitter",
        corrector="twitter",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True, censored=False).tokenize,
        # tokenizer=ws_tokenizer,
        # dicts=[emoticons]
    )

    # using ekphrasis api to alter the initial tweets
    print('starting...')
    # print(data[9582][2])
    for i in range(shift, shift + nb_tweets):
        data[i][2] = " ".join(text_processor.pre_process_doc(data[i][2]))

    for i in range(shift, shift + nb_tweets):
        # data[i][2] = data[i][2].replace("<hashtag>", '')
        # data[i][2] = data[i][2].replace("</hashtag>", '')
        data[i][2] = data[i][2].replace("<allcaps>", '')
        data[i][2] = data[i][2].replace("<elongated>", '')
        data[i][2] = data[i][2].replace("<repeated>", '')
        data[i][2] = data[i][2].replace("<emphasis>", '')
        data[i][2] = data[i][2].replace("<censored>", '')

        data[i][2] = data[i][2].replace("<kiss>", '')
        data[i][2] = data[i][2].replace("<happy>", '')
        data[i][2] = data[i][2].replace("<laugh>", '')
        data[i][2] = data[i][2].replace("<sad>", '')
        data[i][2] = data[i][2].replace("<surprise>", '')
        data[i][2] = data[i][2].replace("<wink>", '')
        data[i][2] = data[i][2].replace("<tong>", '')
        data[i][2] = data[i][2].replace("<annoyed>", '')
        data[i][2] = data[i][2].replace("<seallips>", '')
        data[i][2] = data[i][2].replace("<angel>", '')
        data[i][2] = data[i][2].replace("<devil>", '')
        data[i][2] = data[i][2].replace("<highfive>", '')
        data[i][2] = data[i][2].replace("<heart>", '')

        data[i][2] = data[i][2].replace("<url>", '')
        data[i][2] = data[i][2].replace("<email>", '')
        data[i][2] = data[i][2].replace("<percent>", '')
        data[i][2] = data[i][2].replace("<money>", '')
        data[i][2] = data[i][2].replace("<phone>", '')
        data[i][2] = data[i][2].replace("<user>", '')
        data[i][2] = data[i][2].replace("<time>", '')
        data[i][2] = data[i][2].replace("<date>", '')
        data[i][2] = data[i][2].replace("<number>", '')

        # beautify text
        data[i][2] = " ".join(text_processor.pre_process_doc(data[i][2]))  # ka9e fora megalwnei to censored
        data[i][2] = " ".join(text_processor.pre_process_doc(data[i][2]))
        data[i][2] = data[i][2].replace("<number>", '')
        data[i][2] = data[i][2].replace("<elongated>", '')
        data[i][2] = data[i][2].replace("<date>", '')
        data[i][2] = " ".join(text_processor.pre_process_doc(data[i][2]))

    lexicons_data = []

    for i in range(shift, shift + nb_tweets):
        lexicons_data.append(data[i][2])

    return lexicons_data


def get_labels():
    # nb_tweets = 51075  # number of tweets to read from .txt (max:38791)
    # nb_tweets = 100  # number of tweets to read from .txt (max:38791)
    nb_tweets = 38791  # number of tweets to read from .txt (max:38791) - GET ONLY THE TRAIN VALUES
    shift = 0  # starting point

    # read .txt and make a data array
    with open('input/final_tweets_no_duplicates_no_NAs_plus_TEST.txt', encoding="utf8") as f:
        data = [x.strip().split('\t') for x in f]  # data:489x3

    labels = []
    for i in range(shift, shift + nb_tweets):
        if data[i][1] == 'positive':
            labels.append(0)
        elif data[i][1] == 'neutral':
            labels.append(1)
        elif data[i][1] == 'negative':
            labels.append(2)
        else:
            print('################ error ################')
            print('unexpected label')
            break

    labels = to_categorical(np.asarray(labels))

    return labels


def get_true_labels():
    # nb_tweets = 51075  # number of tweets to read from .txt (max:38791)
    # nb_tweets = 100  # number of tweets to read from .txt (max:38791)
    nb_tweets = 12284  # number of tweets to read from .txt (max:38791) - GET ONLY THE TRAIN VALUES
    shift = 0  # starting point

    # read .txt and make a data array
    with open('input/SemEval2017_true_labels.txt', encoding="utf8") as f:
        data = [x.strip().split('\t') for x in f]  # data:489x3

    labels = []
    for i in range(shift, shift + nb_tweets):
        if data[i][1] == 'positive':
            labels.append(0)
        elif data[i][1] == 'neutral':
            labels.append(1)
        elif data[i][1] == 'negative':
            labels.append(2)
        else:
            print('################ error ################')
            print('unexpected label')
            break

    labels = to_categorical(np.asarray(labels))

    return labels


def get_data(texts, pad, max_nb_words, max_seq_length):
    tokenizer = Tokenizer(num_words=max_nb_words)
    tokenizer.fit_on_texts(texts)  # train on a list of texts
    data = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    # print('Found %s unique tokens.' % len(word_index))

    if pad:
        data = pad_sequences(data, maxlen=max_seq_length)

    return data, word_index


def create_embedding_matrix_glove(word_index, max_nb_words, emb_dim):

    embeddings_index = {}
    try:
        f = open("glove/glove.6B.300d.txt", encoding="utf8")
    except:
        print("Download glove.6B.300d.txt!")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    # prepare embedding matrix
    num_words = min(max_nb_words, len(word_index))
    embedding_matrix = np.zeros((len(word_index) + 1, emb_dim))
    for word, i in word_index.items():
        if i >= max_nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def create_embedding_matrix_w2v(word_index,  limit, max_nb_words, emb_dim):
    num_words = min(max_nb_words, len(word_index))
    embedding_matrix = np.zeros((len(word_index) + 1,emb_dim))
    try:
        if limit:
            word_vectors = KeyedVectors.load_word2vec_format("word2vec/GoogleNews-vectors-negative300.bin",
                                                             binary=True,
                                                             limit=500000)  # C binary format
        else:
            word_vectors = KeyedVectors.load_word2vec_format("word2vec/GoogleNews-vectors-negative300.bin",
                                                             binary=True)  # C binary format
    except:
        print("Download word2vec/GoogleNews-vectors-negative300.bin!")

    for word, i in word_index.items():
        if i >= max_nb_words:
            continue
        try:
            embedding_matrix[i] = word_vectors.wv[word]
        except:
            print("error in word " + word)

    return embedding_matrix


def split_train_test(data_, pad):
    if pad:
        data_train_ = data_[0:38791, :]
        data_test_ = data_[38791:51075, :]
    else:
        data_train_ = data_[0:38791]
        data_test_ = data_[38791:51075]

    return data_train_, data_test_

