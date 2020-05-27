import numpy as np
from nltk.tokenize import WordPunctTokenizer


def clean_data(data):
    n = len(data)
    for i in range(n - 1, -1, -1):
        first_label = WordPunctTokenizer().tokenize(data[i, 2])[0]
        text = WordPunctTokenizer().tokenize(data[i, 1])
        if first_label not in text:
            data = np.delete(data, i, axis=0)
    return data


if __name__ == '__main__':
    SENTENCE = [[1234,
                 "Journey!? Wow... u just became u cooler.  hehe... (is that possible!?)",
                 "just became u cooler",
                 "positive"]]
    S = np.array(SENTENCE)
    S = clean_data(S)
    print(S)
