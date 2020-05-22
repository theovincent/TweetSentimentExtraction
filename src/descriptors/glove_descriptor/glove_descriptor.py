from pathlib import Path
import numpy as np
import pandas as pd
import re
import string
from nltk.tokenize import WordPunctTokenizer


def get_description(word, dictionary, word_size, fill_with, not_seen):
    search = dictionary.iloc[:, 0] == word

    if np.sum(search) > 0:
        return dictionary[search].iloc[0, 1:].to_numpy()
    else:
        if not_seen:
            print(word, ": not seen...")
        return np.ones(word_size) * fill_with


def one_description(tweet, dictionary, sentence_size=15, word_size=50, fill_with=0, not_seen=False):
    # Initialise output
    output = np.ones((sentence_size, word_size)) * fill_with

    # Split the word
    list_words = WordPunctTokenizer().tokenize(tweet)
    nb_words = len(list_words)
    idx_word = 0

    while idx_word < nb_words and idx_word < sentence_size:
        output[idx_word] = get_description(list_words[idx_word], dictionary, word_size, fill_with, not_seen)
        idx_word += 1

    return list_words, np.reshape(output, -1)


def descriptor(tweets, dictionary, sentence_size=15, word_size=50, fill_with=0, not_seen=False):
    nb_tweets = len(tweets)
    # Initialize sets
    x_string = []
    x_scalar = np.ones((nb_tweets, sentence_size * word_size)) * fill_with

    # Describe each tweet
    for idx_tweet in range(nb_tweets):
        description = one_description(tweets[idx_tweet], dictionary, sentence_size, word_size, fill_with, not_seen)
        x_string.append(description[0])
        x_scalar[idx_tweet] = description[1]

    return x_string, x_scalar


if __name__ == "__main__":
    # -- Get the data -- #
    PATH_SAMPLE = Path("../../../data/samples/sample_10_train.csv")
    SAMPLE = pd.read_csv(PATH_SAMPLE)

    # -- Get the descriptor -- #
    PATH_DICTIONARY = Path("../../../data/glove_descriptor/glove.6B.50d.txt")
    DICTIONARY = pd.read_csv(PATH_DICTIONARY, sep=" ", header=None)

    # -- Parameters -- #
    WORD_SIZE = 50  # 50 or 100 or 200 or 300
    SENTENCE_SIZE = 20  # What ever
    FILL_WITH = 0  # If a word is not in the dictionary, [0, ..., 0] will describe it.

    (TWEET_STRING, TWEET_SCALAR) = descriptor(SAMPLE["text"], DICTIONARY, SENTENCE_SIZE, WORD_SIZE, FILL_WITH, not_seen=False)

    # (TWEET_STRING, TWEET_SCALAR) = one_description(SAMPLE["text"][0], DICTIONARY, SENTENCE_SIZE, WORD_SIZE)
    print(TWEET_STRING)
    print("len(TWEET_STRING)", len(TWEET_STRING))
    print(TWEET_SCALAR)
    print("TWEET_SCALAR.shape", TWEET_SCALAR.shape)
