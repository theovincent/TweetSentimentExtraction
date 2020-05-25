# from src.utils.sample_data import SampleData
# from src.utils.post_processing import sentence_to_string


import numpy as np
import re
from nltk.tokenize import WordPunctTokenizer


# NOT USED AT THE MOMENT
def replacement(matchobj):
    if matchobj.group(0) == ' ':
        return ' '
    else:
        return ''


def alphanumeric(text, lower=True):
    new_text = text
    if lower:
        new_text = new_text.lower()
    new_text = re.sub(r'\W+', '', new_text)

    return new_text


def vectorize_string(text, sentence_size=15, word_size=12,
                     split_punctuation=False, alphanumeric_only=True, fill_with="$"):
    if split_punctuation:
        list_words = WordPunctTokenizer().tokenize(text)
    else:
        list_words = text.split()

    if alphanumeric_only:
        to_delete = []
        for i in range(len(list_words)):
            list_words[i] = alphanumeric(list_words[i])
            if list_words[i] == "":
                to_delete.append(i)
        list_words = np.delete(list_words, to_delete)
        print("alphanumeric only \n", list_words)

    filled_text = np.full(sentence_size, fill_with * word_size)
    for i in range(min(sentence_size, len(list_words))):
        s = len(list_words[i])
        if s < word_size:
            filled_text[i] = list_words[i] + filled_text[i]
        else:
            filled_text[i] = list_words[i][:word_size]

    return filled_text


def descriptor(list_words, alphanumeric_only=True):
    inputs = list_words
    outputs = []

    for word in inputs:
        output = []
        for character in word:
            number = ord(character)
            if alphanumeric_only:
                number -= 96
            output.append(number)
        outputs.append(output)

    return np.array(outputs)


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


def convert_labels(data, sentence_size, split_punctuation=True):
    y = []
    for tweet in data:
        if split_punctuation:
            sentence = WordPunctTokenizer().tokenize(tweet[1])
            label = WordPunctTokenizer().tokenize(tweet[2])
        else:
            sentence = tweet[1].split()
            label = tweet[2].split()

        label = convert_label(np.array(sentence), np.array(label), sentence_size)
        y.append(label)
    return np.array(y)


if __name__ == "__main__":

    ALPHANUM_ONLY = False
    WORD_SIZE = 12
    SENTENCE_SIZE = 25
    FILL_WITH = "$"
    SPLIT_PUNCTUATION = False

    SENTENCE = "Journey!? Wow... u just became u cooler.  hehe... (is that possible!?)"

    LABEL = "!? Wow... u"

    print("Input : \n", SENTENCE)

    print("Label :")
    print(convert_labels([("", SENTENCE, LABEL, "")], SENTENCE_SIZE))

    SENTENCE = vectorize_string(SENTENCE, alphanumeric_only=ALPHANUM_ONLY,
                                sentence_size=SENTENCE_SIZE,
                                word_size=WORD_SIZE,
                                split_punctuation=SPLIT_PUNCTUATION,
                                fill_with=FILL_WITH
                                )

    LABEL = vectorize_string(LABEL, alphanumeric_only=ALPHANUM_ONLY,
                             sentence_size=SENTENCE_SIZE,
                             word_size=WORD_SIZE,
                             split_punctuation=SPLIT_PUNCTUATION,
                             fill_with=FILL_WITH
                             )

    print("In a fixed size matrix :")
    print(SENTENCE)

    SENTENCE = descriptor(SENTENCE, alphanumeric_only=ALPHANUM_ONLY)
    print("Output :")
    print(SENTENCE)
