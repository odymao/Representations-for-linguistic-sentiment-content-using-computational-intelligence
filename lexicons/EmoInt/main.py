'''
first run: python setup.py install
Run in python 2.7
'''

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def split_train_test(data_):
    data_train_ = data_[0:38791, :]
    data_test_ = data_[38791:51075, :]
    return data_train_, data_test_

from tweetokenize.tweetokenize.tokenizer import Tokenizer
from emoint.featurizers.emoint_featurizer import EmoIntFeaturizer

# Load raw_data for lexicons use
lexicon_data_file = Path("../../input/lexicons_data.npy")

if lexicon_data_file.is_file():
    tweets = np.load(lexicon_data_file)
else:
    print("You have to first run data/main.py to generate lexicons_data.npy")


featurizer = EmoIntFeaturizer()
tokenizer = Tokenizer()

features = []

# Extract lexicon's features
for tweet in tweets:
    features.append(featurizer.featurize(tweet, tokenizer))


# Split data to train and test
f_train, f_test = split_train_test(features)

# Create output files
print("Creating lexicons_TRAIN and lexicons_TEST into /outputs folder")
np.save("../../outputs/lexicons_TRAIN", f_train)
np.save("../../outputs/lexicons_TEST", f_test)

