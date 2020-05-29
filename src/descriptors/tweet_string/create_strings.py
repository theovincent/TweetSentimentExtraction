from pathlib import Path
import numpy as np
import pandas as pd


def create_strings(tweet_originals, tokenizer, sentence_size):
    nb_tweets = len(tweet_originals)

    # Initialize strings
    tweet_strings = ["tweet"] * nb_tweets

    for idx_tweet in range(nb_tweets):
        tweet_strings[idx_tweet] = list(map(np.str_, tokenizer.tokenize(tweet_originals[idx_tweet], sentence_size)))

    return tweet_strings


if __name__ == "__main__":
    from src.descriptors.tokenizer.tokenizer import Tokenizer

    # -- Get the data -- #
    PATH_SAMPLE = Path("../../../data/samples/sample_1000_train.csv")
    SAMPLE = pd.read_csv(PATH_SAMPLE).to_numpy()

    # -- Get the original tweets -- #
    TWEET_ORIGINALS = SAMPLE[:, 1]
    print(TWEET_ORIGINALS[19])
    # -- Define the tokenizer -- #
    TOKENIZER = Tokenizer()

    SENTENCE_SIZE = 50

    print(create_strings(TWEET_ORIGINALS, TOKENIZER, SENTENCE_SIZE)[19])
