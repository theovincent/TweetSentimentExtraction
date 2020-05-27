from pathlib import Path
import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer


def find_word(word, dictionary, additional_dic):
    word = word.lower()

    if word in additional_dic:
        word = additional_dic[word]

    search = dictionary.iloc[:, 0] == word

    return np.sum(search) > 0


def separation(tweet, tokenizer):
    word = tokenizer.tokenize(tweet)
    return word


def match_percent(tweets, dictionary, additional_dic):
    matches = 0

    # Initialize tokenizer
    tokenizer = TweetTokenizer(strip_handles=True)

    for tweet in tweets:
        matches_tweet = 0
        words = separation(tweet, tokenizer)
        for word in words:
            find = find_word(word, dictionary, additional_dic)
            if not find:
                print(word)
            matches_tweet += find
        matches += matches_tweet / len(words)

    return matches / len(tweets)


if __name__ == "__main__":
    # -- Get the data -- #
    PATH_SAMPLE = Path("../../../data/samples/sample_1000_train.csv")
    SAMPLE = pd.read_csv(PATH_SAMPLE).to_numpy()

    # -- Get the tweet_scalar_glove -- #
    PATH_DICTIONARY = Path("../../../data/descriptor_glove/glove.6B.50d.txt")
    DICTIONARY = pd.read_csv(PATH_DICTIONARY, sep=" ", header=None)

    ADDITIONAL_DIC = {"..": "...", "<3": "love"}

    # -- Get the text -- #
    TWEETS = SAMPLE[:, 1]

    PERCENT_MATCH = match_percent(TWEETS, DICTIONARY, ADDITIONAL_DIC)
    print("The percentage of matches is :", PERCENT_MATCH)

    # Low case : 95 % percent of match with 1000 samples
    # Additional dictionnary + Tweet : 96 %
