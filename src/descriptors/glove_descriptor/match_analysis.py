from pathlib import Path
import pandas as pd
import numpy as np
from nltk.tokenize import WordPunctTokenizer


def find_word(word, dictionary):
    # if word == ":
    word = word.lower()
    search = dictionary.iloc[:, 0] == word

    return np.sum(search) > 0


def separation(tweet):
    return WordPunctTokenizer().tokenize(tweet)


def match_percent(tweets, dictionary):
    matches = 0
    for tweet in tweets:
        matches_tweet = 0
        words = separation(tweet)
        for word in words:
            matches_tweet += find_word(word, dictionary)
        matches += matches_tweet / len(words)

    return matches / len(tweets)


if __name__ == "__main__":
    # -- Get the data -- #
    PATH_SAMPLE = Path("../../../data/samples/sample_10_train.csv")
    SAMPLE = pd.read_csv(PATH_SAMPLE).to_numpy()

    # -- Get the descriptor -- #
    PATH_DICTIONARY = Path("../../../data/glove_descriptor/glove.6B.50d.txt")
    DICTIONARY = pd.read_csv(PATH_DICTIONARY, sep=" ", header=None)

    # -- Get the text -- #
    TWEETS = SAMPLE[:, 1]

    PERCENT_MATCH = match_percent(TWEETS, DICTIONARY)
    print("The percentage of matches is :", PERCENT_MATCH)
