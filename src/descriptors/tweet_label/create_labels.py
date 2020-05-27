from pathlib import Path
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer


def convert_label(sentence, label, sentence_size):
    new_label = np.full(len(sentence), 0)
    n = len(label)
    possibles = np.where(sentence == label[0])[0]

    sol = 0
    for p in possibles:
        check = sentence[p: p + n]
        if np.all(check == label):
            sol = p
            break

    new_label[sol: sol + n] = 1
    new_label = np.concatenate((new_label, np.zeros(max(0, sentence_size - len(sentence)))))

    return new_label


def create_labels(tweet_string, tweet_string_label):
    y = np.zeros((len(data), sentence_size))

    tokenizer = TweetTokenizer(strip_handles=True)

    i = -1
    for tweet in data:
        i += 1
        if split_punctuation:
            sentence = tokenizer.tokenize(tweet[1])
            label = tokenizer.tokenize(tweet[2])
        else:
            sentence = tweet[1].split()
            label = tweet[2].split()

        label = convert_label(np.array(sentence), np.array(label), sentence_size)

        y[i] = label
    return np.array(y)


if __name__ == "__main__":
    # -- Get the data -- #
    PATH_SAMPLE = Path("../../../data/samples/sample_10_train.csv")
    SAMPLE = pd.read_csv(PATH_SAMPLE).to_numpy()

    IDX = 9

    print(SAMPLE[IDX])
    print()
    print(create_labes([SAMPLE[IDX]], 50))
