from src.utils.sample_data import SmallData
from src.utils.post_processing import sentance_to_string


import numpy as np
import pandas as pd
import os
import re


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


def vectorize_string(text, sentence_size=15, word_size=12, alphanumeric_only=True):
    list_text = text.split()

    if alphanumeric_only:
        to_delete = []
        for i in range(len(list_text)):
            list_text[i] = alphanumeric(list_text[i])
            if list_text[i] == "":
                to_delete.append(i)
        list_text = np.delete(list_text, to_delete)
        print("alphanumeric only \n", list_text)

    filled_text = np.full(sentence_size, "$" * word_size)
    for i in range(min(sentence_size, len(list_text))):
        s = len(list_text[i])
        if s < word_size:
            filled_text[i] = list_text[i] + filled_text[i]
        else:
            filled_text[i] = list_text[i][:word_size]

    return filled_text


def descriptor(list_text, alphanumeric_only=True):
    inputs = list_text
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


if __name__ == "__main__":

    ALPHANUM_ONLY = False
    SENTENCE = "Journey!? Wow... u just became cooler.  hehe... (is that possible!?)"
    print("Input : \n", SENTENCE)

    SENTENCE = vectorize_string(SENTENCE, alphanumeric_only=ALPHANUM_ONLY)
    print("In a fixed size matrix :")
    print(SENTENCE)

    SENTENCE = descriptor(SENTENCE, alphanumeric_only=ALPHANUM_ONLY)
    print("Output :")
    print(SENTENCE)

