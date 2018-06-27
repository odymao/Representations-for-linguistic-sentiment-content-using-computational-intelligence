# Load .mat files from features and scale them

import numpy as np
import scipy.io
from sklearn.preprocessing import normalize, MinMaxScaler


def main():


    # ## Load outliered features from matlab files
    # # Load .mat files
    # features_train_norm = scipy.io.loadmat("input/features_train_outliered.mat")
    # features_test_norm = scipy.io.loadmat("input/features_test_outliered.mat")
    #
    # features_train_norm = features_train_norm.get("data")
    # features_test_norm = features_test_norm.get("data")


    # Load arithmetic features without being outliered
    try:
        features_train = np.load("input/features_TRAIN.npy")
        features_test = np.load("input/features_TEST.npy")

        # Scale features
        min_max_scaler = MinMaxScaler()
        features_train_norm = min_max_scaler.fit_transform(features_train)
        features_test_norm = min_max_scaler.fit_transform(features_test)

        # Create output files
        np.save("outputs/features_TRAIN_norm_ourliered", features_train_norm)
        np.save("outputs/features_TEST_norm_ourliered", features_test_norm)
    except:
        print("You have to first run POS tagging to generate arithmetic features")



if __name__ == "__main__":
    print("Creating features files")
    main()
