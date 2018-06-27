from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from nltk import word_tokenize, pos_tag
from collections import Counter
from itertools import chain, repeat, islice
import numpy as np


def get_raw_data():
    # nb_tweets = 51075  # number of tweets to read from .txt (max:38791)
    # nb_tweets = 100  # number of tweets to read from .txt (max:38791)
    nb_tweets = 1  # number of tweets to read from .txt (max:38791)
    shift = 0  # starting point
    nb_annotations = 7  # number of annotations counted to form the aritmetic features

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

    #initializing array for hosting the arithmetic Features
    data_arithmetic_features = np.empty((nb_tweets, nb_annotations), dtype=int)

    # using ekphrasis api to alter the initial tweets
    print('starting...')
    # print(data[9582][2])
    for i in range(shift, shift + nb_tweets):
        data[i][2] = " ".join(text_processor.pre_process_doc(data[i][2]))

    for i in range(shift, shift + nb_tweets):

        # collecting the arithmetic features
        data_arithmetic_features[i][0] = str(data[i][2].count("<hashtag>"))
        data_arithmetic_features[i][1] = str(data[i][2].count("<allcaps>"))
        data_arithmetic_features[i][2] = str(data[i][2].count("<elongated>"))
        data_arithmetic_features[i][3] = str(data[i][2].count("<repeated>"))
        data_arithmetic_features[i][4] = str(data[i][2].count("<emphasis>"))
        data_arithmetic_features[i][5] = str(data[i][2].count("<censored>"))
        data_arithmetic_features[i][6] = str(data[i][2].count("<kiss>") +
                                             data[i][2].count("<happy>") +
                                             data[i][2].count("<laugh>") +
                                             data[i][2].count("<sad>") +
                                             data[i][2].count("<surprise>") +
                                             data[i][2].count("<wink>") +
                                             data[i][2].count("<tong>") +
                                             data[i][2].count("<annoyed>") +
                                             data[i][2].count("<seallips>") +
                                             data[i][2].count("<angel>") +
                                             data[i][2].count("<devil>") +
                                             data[i][2].count("<highfive>") +
                                             data[i][2].count("<heart>"))


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

    ################### Features ###################
    ## will be used in the features input

    # Split features to train and test
    features_train, features_test = split_train_test(data_arithmetic_features)

    # Create output features files
    np.save("input/features_TRAIN", features_train)
    np.save("input/features_TEST", features_test)

    ################################################

    return raw_data


def get_pos(data):

    #### POS Tagging meanings ####
    # ADJ	adjective	new, good, high, special, big, local
    # ADP	adposition	on, of, at, with, by, into, under
    # ADV	adverb	really, already, still, early, now
    # CONJ	conjunction	and, or, but, if, while, although
    # DET	determiner, article	the, a, some, most, every, no, which
    # NOUN	noun	year, home, costs, time, Africa
    # NUM	numeral	twenty-four, fourth, 1991, 14:24
    # PRT	particle	at, on, out, over per, that, up, with
    # PRON	pronoun	he, their, her, its, my, I, us
    # VERB	verb	is, say, told, given, playing, would
    # .	punctuation marks	. , ; !
    # X	other	ersatz, esprit, dunno, gr8, univeristy
    
    pos_sequence_matrix = []
    n = 50  # for padding
    for line in data:

        currentTextLine = word_tokenize(line)
        taggedText = pos_tag(currentTextLine, tagset="universal")

        count = Counter([j for i, j in taggedText])
        dict_count = dict(count)

        dict_taggedSentence = dict(taggedText)
        print(dict_taggedSentence)
        pos_Sequence = []
        # pos_Tags = {}
        for key, value in dict_taggedSentence.items():
            # pos_Tags.update({value: key})

            if value == 'ADJ':
                pos_Sequence.append(1)
            elif value == 'ADP':
                pos_Sequence.append(2)
            elif value == 'ADV':
                pos_Sequence.append(3)
            elif value == 'CONJ':
                pos_Sequence.append(4)
            elif value == 'DET':
                pos_Sequence.append(5)
            elif value == 'NOUN':
                pos_Sequence.append(6)
            elif value == 'NUM':
                pos_Sequence.append(7)
            elif value == 'PRT':
                pos_Sequence.append(8)
            elif value == 'PRON':
                pos_Sequence.append(9)
            elif value == 'VERB':
                pos_Sequence.append(10)
            elif value == '.':
                pos_Sequence.append(11)
            elif value == 'X':
                pos_Sequence.append(12)
            else:
                print("Unknown input")
                break

        pos_sequence_matrix.append(list(pad(pos_Sequence, n, 0)))

    pos_sequence_matrix = np.array(pos_sequence_matrix)


    return pos_sequence_matrix


def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))


def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)


def get_hot_vector(line, index):
    line[index] = 1
    return line


def get_bin_vector(index):

    if index == 0:
        vec = [0, 0, 0, 0]
    elif index == 1:
        vec = [0, 0, 0, 1]
    elif index == 2:
        vec = [0, 0, 1, 0]
    elif index == 3:
        vec = [0, 0, 1, 1]
    elif index == 4:
        vec = [0, 1, 0, 0]
    elif index == 5:
        vec = [0, 1, 0, 1]
    elif index == 6:
        vec = [0, 1, 1, 0]
    elif index == 7:
        vec = [0, 1, 1, 1]
    elif index == 8:
        vec = [1, 0, 0, 0]
    elif index == 9:
        vec = [1, 0, 0, 1]
    elif index == 10:
        vec = [1, 0, 1, 0]
    elif index == 11:
        vec = [1, 0, 1, 1]
    elif index == 12:
        vec = [1, 1, 0, 0]
    else:
        print("Error in pos index")


    return vec


def embedding_hot():

    emb_matrix = np.zeros((13, 13), dtype=int)

    for i in range(13):
        emb_matrix[i] = get_hot_vector(emb_matrix[i], i)

    # np.save("pos_emb_matrix", emb_matrix)
    return emb_matrix


def embedding_bin():

    emb_matrix = np.zeros((13, 4), dtype=int)

    for i in range(13):
        emb_matrix[i] = get_bin_vector(i)

    # np.save("pos_emb_matrix", emb_matrix)
    return emb_matrix


def split_train_test(data_):
    data_train_ = data_[0:38791, :]
    data_test_ = data_[38791:51075, :]
    return data_train_, data_test_

