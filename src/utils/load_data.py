import pandas as pd
from pathlib import Path
import numpy as np

from src.utils.data_descriptor import descriptor
from src.utils.data_descriptor import vectorize_string
from src.utils.data_descriptor import convert_labels


def load_data(data, alphanum_only, word_size, sentence_size, fill_with):

    # list of the tweets, at the right shape, with "$" to fill
    x_string = []

    # data to use in the classifier
    # the last element (or the lasts, if feeling_weight > 1) of each line is the feeling
    x_scalar = []

    # labels : each element is an array of size SENTENCE_SIZE of 0 and 1
    y = []

    # to repeat the feeling several times and then put a weight on the feeling
    feeling_weight = 1

    for tweet in data:
        sentence = vectorize_string(tweet[1], alphanumeric_only=alphanum_only,
                                    sentence_size=sentence_size,
                                    word_size=word_size,
                                    fill_with=fill_with
                                    )

        label = vectorize_string(tweet[2], alphanumeric_only=alphanum_only,
                                 sentence_size=sentence_size,
                                 word_size=word_size,
                                 fill_with=fill_with
                                 )

        x_string.append(sentence)

        label = convert_labels(sentence, label)
        y.append(label)

        sentence = descriptor(sentence, alphanumeric_only=alphanum_only).flatten()
        x = np.concatenate((sentence, [tweet[3]] * feeling_weight))
        x_scalar.append(x)

    x_scalar = np.array(x_scalar)
    x_string = np.array(x_string)
    y = np.array(y)

    return x_string, x_scalar, y


if __name__ == '__main__':
    CSV_NAME = "sample_100.csv"
    CSV_PATH = Path("../../data/samples/") / CSV_NAME
    CSV_DATA = pd.read_csv(CSV_PATH)
    DATA = pd.DataFrame.to_numpy(CSV_DATA)

    ALPHANUM_ONLY = False
    WORD_SIZE = 12
    SENTENCE_SIZE = 15
    FILL_WITH = "$"

    X_STRING, X_SCALAR, Y = load_data(DATA, ALPHANUM_ONLY, WORD_SIZE, SENTENCE_SIZE, FILL_WITH)

    print(DATA.shape, X_STRING.shape, Y.shape, X_SCALAR.shape)

    print(DATA[0])
    print(X_STRING[0])
    print(Y[0])
    print(X_SCALAR[0])
