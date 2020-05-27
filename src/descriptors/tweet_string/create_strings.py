from pathlib import Path
import numpy as np
import pandas as pd


def create_strings(tweet_original, tokenizer):
    nb_tweets = len(tweet_original)

    # Initialize strings
    tweet_string = ["tweet"] * nb_tweets

    for idx_tweet in range(nb_tweets):
        tweet_string[idx_tweet] = tokenizer.tokenize(tweet_original[idx_tweet])

    return tweet_string


if __name__ == "__main__":
    from src.descriptors.tokenizer.tokenizer import Tokenizer

    # -- Get the data -- #
    PATH_SAMPLE = Path("../../../data/samples/sample_10_train.csv")
    SAMPLE = pd.read_csv(PATH_SAMPLE).to_numpy()

    # -- Get the original tweets -- #
    TWEET_ORIGINAL = SAMPLE[:, 1]

    # -- Define the tokenizer -- #
    TOKENIZER = Tokenizer()

    print(create_strings(TWEET_ORIGINAL, TOKENIZER))
