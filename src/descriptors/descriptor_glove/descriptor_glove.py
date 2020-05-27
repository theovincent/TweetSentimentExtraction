from pathlib import Path
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer


def get_description(word, dictionary, additional_dic, word_size, fill_with, not_seen):
    # if word == ":
    word = word.lower()

    if word in additional_dic:
        word = additional_dic[word]

    search = dictionary.iloc[:, 0] == word

    if np.sum(search) > 0:
        return dictionary[search].iloc[0, 1:].to_numpy()
    else:
        if not_seen:
            print(word, ": not seen...")
        return np.ones(word_size) * fill_with


def one_description(tweet, tokenizer, dictionary, additional_dic, sentence_size, word_size, fill_with, not_seen):
    # Initialise output
    output = np.ones((sentence_size, word_size)) * fill_with

    # Split the word
    list_words = tokenizer.tokenize(tweet)
    nb_words = len(list_words)
    idx_word = 0

    while idx_word < nb_words and idx_word < sentence_size:
        output[idx_word] = get_description(list_words[idx_word], dictionary, additional_dic, word_size, fill_with, not_seen)
        idx_word += 1

    return list_words, np.reshape(output, -1)


def tweet_scalar_glove(tweet_string, sentiments, dictionary, additional_dic, options):
    # Get the options
    (word_size, sentence_size, fill_with, sentiment_weight) = options

    # Get the tweets and the sentiments
    tweets = samples[:, 1]
    sentiment = samples[:, 3]

    nb_tweets = len(tweets)
    # Initialize sets
    tweet_string = []
    tweet_scalar = np.ones((nb_tweets, sentence_size * word_size + 1)) * fill_with

    # Initialize tokenizer
    tokenizer = TweetTokenizer(strip_handles=True)

    # Describe each tweet
    for idx_tweet in range(nb_tweets):
        # Describe the tweet
        description = one_description(tweets[idx_tweet], tokenizer, dictionary, additional_dic, sentence_size, word_size, fill_with, not_seen)

        # Add the sentiment
        sentiment_description = np.concatenate((description[1], np.array([sentiment[idx_tweet] * sentiment_weight])))

        # Update lists
        tweet_string.append(description[0])
        tweet_scalar[idx_tweet] = sentiment_description

    return np.array(tweet_string, dtype=str), tweet_scalar


if __name__ == "__main__":
    # -- Get the data -- #
    PATH_SAMPLE = Path("../../../data/samples/sample_10_train.csv")
    SAMPLE = pd.read_csv(PATH_SAMPLE).to_numpy()

    # -- Get the tweet_scalar_glove -- #
    PATH_DICTIONARY = Path("../../../data/descriptor_glove/glove.6B.50d.txt")
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

    print(TWEET_STRING)
    print("len(TWEET_STRING)", len(TWEET_STRING))
    print(TWEET_SCALAR)
    print("TWEET_SCALAR.shape", TWEET_SCALAR.shape)
