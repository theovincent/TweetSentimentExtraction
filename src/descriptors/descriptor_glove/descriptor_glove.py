import numpy as np


def get_description(word, dictionary, additional_dic, vector_to_fill):
    # Modify the word to maximize the probability to find the word in the dictionary
    word = word.lower()

    if word in additional_dic:
        word = additional_dic[word]

    # Search the word in the dictionary
    search = dictionary.iloc[:, 0] == word

    if np.sum(search) > 0:
        return dictionary[search].iloc[0, 1:].to_numpy()
    else:
        return vector_to_fill


def one_description(tweet_string, dictionary, additional_dic, sentence_size, word_size, fill_with):
    # Initialise the description
    description = np.ones((sentence_size, word_size)) * fill_with

    # Initialise the vector to put when the word is not find
    vector_to_fill = np.ones(word_size) * fill_with

    nb_words = len(tweet_string)
    idx_word = 0

    while idx_word < nb_words and idx_word < sentence_size:
        description[idx_word] = get_description(tweet_string[idx_word], dictionary, additional_dic, vector_to_fill)
        idx_word += 1

    return np.reshape(description, -1)


def tweet_scalar_glove(tweet_strings, sentiments, dictionary, additional_dic, options):
    nb_tweets = len(tweet_strings)

    # Get the options
    (word_size, sentence_size, fill_with, sentiment_weight) = options

    # Initialize tweet scalar
    tweet_scalar = np.ones((nb_tweets, sentence_size * word_size + 1)) * fill_with  # +1 for the sentiment

    # Create the vector to fill if

    # Describe each tweet
    for idx_tweet in range(nb_tweets):
        # Describe the tweet
        description = one_description(tweet_strings[idx_tweet], dictionary, additional_dic, sentence_size, word_size, fill_with)

        # Add the sentiment
        description_sentiment = np.concatenate((description, np.array([sentiments[idx_tweet] * sentiment_weight])))

        # Update lists
        tweet_scalar[idx_tweet] = description_sentiment

    return tweet_scalar


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd
    # -- Get the data -- #
    PATH_SAMPLE = Path("../../../data/samples/sample_10_train.csv")
    SAMPLE = pd.read_csv(PATH_SAMPLE).to_numpy()

    # -- Parameters -- #
    WORD_SIZE = 50  # 50 or 100 or 200 or 300
    FILL_WITH = 0  # If a word is not in the dictionary, [0, ..., 0] will describe it.
    SENTIMENT_WEIGHT = 2  # Multiply the sentiment by a factor
    SENTENCE_SIZE = 20  # What ever
    OPTIONS = [WORD_SIZE, SENTENCE_SIZE, FILL_WITH, SENTIMENT_WEIGHT]

    # -- Get the original tweets -- #
    TWEET_ORIGINALS = SAMPLE[:, 1]

    # -- Get the decomposition of the string -- #
    from src.descriptors.tweet_string.create_strings import create_strings
    from src.descriptors.tokenizer.tokenizer import Tokenizer
    TOKENIZER = Tokenizer()
    TWEET_STRINGS = create_strings(TWEET_ORIGINALS, TOKENIZER, SENTENCE_SIZE)

    # -- Get the tweet_scalar_glove -- #
    PATH_DICTIONARY = Path("../../../data/glove_descriptor/glove.6B.50d.txt")
    DICTIONARY = pd.read_csv(PATH_DICTIONARY, sep=" ", header=None)

    # Additional dictionary
    ADDITIONAL_DIC = {"..": "...", "<3": "love"}

    # Get the sentiments
    SENTIMENTS = SAMPLE[:, -1]

    TWEET_SCALARS = tweet_scalar_glove(TWEET_STRINGS, SENTIMENTS, DICTIONARY, ADDITIONAL_DIC, OPTIONS)

    print(TWEET_STRINGS)
    print("len(TWEET_STRINGS)", len(TWEET_STRINGS))
    print(TWEET_SCALARS)
    print("TWEET_SCALAR.shape", TWEET_SCALARS.shape)
