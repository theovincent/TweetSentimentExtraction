import pandas as pd
from pathlib import Path
import numpy as np

from descriptors.descriptor_ascii.data_descriptor import descriptor
from descriptors.descriptor_ascii.data_descriptor import vectorize_string


def load_data(x_sentence, word_size, sentence_size, fill_with, feeling_weight=1, feelings=None):

    # list of the tweets, at the right shape, with "$" to fill
    x_string = []

    # data to use in the classifier
    # the last element (or the lasts, if feeling_weight > 1) of each line is the feeling
    x_scalar = []

    # to repeat the feeling several times and then put a weight on the feeling

    for i, sentence in enumerate(x_sentence):
        filled_sentence = vectorize_string(sentence,
                                           sentence_size=sentence_size,
                                           word_size=word_size,
                                           fill_with=fill_with
                                           )

        x_string.append(filled_sentence)

        ascii_sentence = descriptor(filled_sentence, False).flatten()

        if feelings is not None:
            x = np.concatenate((ascii_sentence, [feelings[i] * feeling_weight]))
        else:
            x = np.array(ascii_sentence)

        x_scalar.append(x)

    x_scalar = np.array(x_scalar)
    x_string = np.array(x_string)

    return x_string, x_scalar


if __name__ == '__main__':
    """
    CSV_NAME = "sample_100.csv"
    CSV_PATH = Path("../../data/samples/") / CSV_NAME
    CSV_DATA = pd.read_csv(CSV_PATH)
    DATA = pd.DataFrame.to_numpy(CSV_DATA)

    ALPHANUM_ONLY = False
    WORD_SIZE = 12
    SENTENCE_SIZE = 15
    FILL_WITH = "$"
    SPLIT_PUNCTUATION = False

    X_STRING, X_SCALAR = load_data(DATA, WORD_SIZE, SENTENCE_SIZE, FILL_WITH, SPLIT_PUNCTUATION, ALPHANUM_ONLY)

    print(DATA.shape, X_STRING.shape, X_SCALAR.shape)

    print(DATA[0])
    print(X_STRING[0])
    print(X_SCALAR[0])
    """
