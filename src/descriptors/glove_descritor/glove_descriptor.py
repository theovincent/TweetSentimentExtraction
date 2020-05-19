from pathlib import Path
import numpy as np
import pandas as pd
import re
import string
from nltk.tokenize import WordPunctTokenizer


def alphanumeric(text, lower=True):
    new_text = text
    if lower:
        new_text = new_text.lower()
    new_text = re.sub(r'\W+', '', new_text)

    return new_text


def get_description(word, descriptions, word_size, fill_with):
    search = descriptions.iloc[:, 0] == word

    if np.sum(search) > 0:
        return descriptions[search].iloc[0, 1:].to_numpy()
    else:
        print("Not seen...")
        return np.ones(word_size) * fill_with


def descriptor(tweet, descriptions, sentence_size=15, word_size=50, alphanumeric_only=True, fill_with=0):
    # Initialise output
    output = np.ones((sentence_size, word_size)) * fill_with

    # Split the word
    list_words = WordPunctTokenizer().tokenize(tweet)
    nb_words = len(list_words)
    idx_word = 0

    while idx_word < nb_words and idx_word < sentence_size:
        print(list_words[idx_word])
        output[idx_word] = get_description(list_words[idx_word], descriptions, word_size, fill_with)
        idx_word += 1

    return output


if __name__ == "__main__":
    # -- Get the data -- #
    PATH_SAMPLE = Path("../../../data/samples/sample_100.csv")
    SAMPLE = pd.read_csv(PATH_SAMPLE)

    # -- Get the descriptor -- #
    PATH_DESCRIPTOR = Path("../../../data/glove_descriptor/glove.6B.50d.txt")
    DESCRIPTIONS = pd.read_csv(PATH_DESCRIPTOR, sep=" ", header=None)

    # -- Parameters -- #
    WORD_SIZE = 50  # 50 or 100 or 200 or 300
    SENTENCE_SIZE = 15  # What ever
    FILL_WITH = 0  # If a word is not in the descriptions, [0, ..., 0] will describe it.

    TWEET = SAMPLE["text"][0]
    print(TWEET)
    OUTPUT = descriptor(TWEET, DESCRIPTIONS, SENTENCE_SIZE, WORD_SIZE, alphanumeric_only=False)
    print(OUTPUT)
