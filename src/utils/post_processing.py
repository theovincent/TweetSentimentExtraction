import numpy as np


def pred_to_sentence(original_strings, predictions):
    return np.where(
        predictions >= 0.5,
        original_strings,
        np.full(original_strings.shape, "$")
    )


def sentance_to_string(sentence):
    return " ".join(sentence)


def filter_character(string, character):
    return string.translate({character: None})
