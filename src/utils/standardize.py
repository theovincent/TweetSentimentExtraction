from pathlib import Path
import numpy as np
import pandas as pd


def standardize(descriptions):
    descriptions[:, : -1] -= np.mean(descriptions[:, : -1], axis=0)
    descriptions[:, : -1] /= (np.std(descriptions[:, : -1], axis=0) + 10**-16)


if __name__ == "__main__":
    from src.descriptors.descriptor_glove.descriptor_glove import tweet_scalar_glove

    # -- Get the data -- #
    PATH_SAMPLE = Path("../../data/samples/sample_10_train.csv")
    SAMPLE = pd.read_csv(PATH_SAMPLE).to_numpy()

    # -- Get the tweet_scalar_glove -- #
    PATH_DICTIONARY = Path("../../data/descriptor_glove/glove.6B.50d.txt")
    DICTIONARY = pd.read_csv(PATH_DICTIONARY, sep=" ", header=None)

    # -- Additional dictionary -- #
    ADDITIONAL_DIC = {"..": "...", "<3": "love"}

    # -- Parameters -- #
    WORD_SIZE = 50  # 50 or 100 or 200 or 300
    SENTENCE_SIZE = 20  # What ever
    FILL_WITH = 0  # If a word is not in the dictionary, [0, ..., 0] will describe it.
    SENTIMENT_WEIGHT = 2  # Multiply the sentiment by a factor
    OPTIONS = [WORD_SIZE, SENTENCE_SIZE, FILL_WITH, SENTIMENT_WEIGHT]

    (TWEET_STRING, TWEET_SCALAR) = tweet_scalar_glove(SAMPLE, DICTIONARY, ADDITIONAL_DIC, OPTIONS, not_seen=False)

    standardize(TWEET_SCALAR)
    print(TWEET_STRING)
    print("len(TWEET_STRING)", len(TWEET_STRING))
    print(TWEET_SCALAR)
    print("TWEET_SCALAR.shape", TWEET_SCALAR.shape)
