from pathlib import Path
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer


def convert_label(sentence, label, sentence_size):
    new_label = np.zeros(sentence_size)

    n = len(label)
    possibles = np.where(sentence == label[0])[0]

    sol = 0
    for p in possibles:
        check = sentence[p: p + n]
        if np.all(check == label):
            sol = p
            break

    new_label[sol: sol + n] = 1
    # new_label = np.concatenate((new_label, np.zeros(max(0, sentence_size - len(sentence)))))
    return new_label


def create_labels(x_sentence, y_string, sentence_size):
    """Returns : y : array where each element is an array of size SENTENCE_SIZE of 0 and 1"""

    n = len(x_sentence)
    y = np.zeros((n, sentence_size))

    # tokenizer = TweetTokenizer(strip_handles=True)

    for i in range(n):
        label = convert_label(np.array(x_sentence[i]), np.array(y_string[i]), sentence_size)
        y[i] = label
    return np.array(y)


if __name__ == "__main__":
    EX = ["i", "feel", "really", "bored", "."]
    EX_LABEL = ["bored", "."]

    print(EX)
    print(EX_LABEL)
    print()
    print(create_labels([EX], [EX_LABEL], 50))
