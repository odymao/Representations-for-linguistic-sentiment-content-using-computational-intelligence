from stanfordcorenlp import StanfordCoreNLP

import json
import numpy as np

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'sentiment',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens


def get_corenlp(data):
    sNLP = StanfordNLP()

    sents = []
    c = 0
    for tweet in data:
        c += 1  # counter for visualization
        if c % 1000 == 0:
            print(c)

        result = sNLP.annotate(tweet)
        sents.append(result.get("sentences"))

    return sents


def get_sents(data):
    sentiments = []

    # iterate through the sentences of the tweet
    for tweet in data:
        dists = []
        for sentence in tweet:
            currentDist = sentence.get("sentimentDistribution")
            dists.append(currentDist)
        sentiments.append(dists)

    return sentiments


def gen_empty_dists(n):
    empty = [0, 0, 0, 0, 0]
    temp = []
    for i in range(n):
        temp.append(empty)

    return temp


def pad_tweets(data, length):
    # length: the length that the tweet will be padded to
    # 5: the default number of sentiment distribution per tweet
    sentiment_dist = 5

    padded_data = np.zeros((len(data), length, sentiment_dist))
    for i in range(len(data)):

        if len(data[i]) >= length:
            temp = data[i][0:length]
            padded_data[i, :] = np.asarray(temp)
        else:
            padded_data[i, :] = np.concatenate((data[i], gen_empty_dists(length - len(data[i]))))

    return padded_data


def split_train_test(data_):
    data_train_ = data_[0:38791, :]
    data_test_ = data_[38791:51075, :]
    return data_train_, data_test_
