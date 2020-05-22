import numpy as np


def add_spaces(original_string, original_sentence):
    while original_string[0] == " ":
        original_string = original_string[1:]

    spaces = np.zeros(len(original_sentence), dtype=bool)

    current_index = 0
    word_count = -1

    for word in original_sentence:
        filtered_word = filter_character(word, "$")

        current_index += len(filtered_word)
        word_count += 1

        if current_index < len(original_string) and original_string[current_index] == " ":
            spaces[word_count] = True

            while original_string[current_index] == " ":
                current_index += 1

    return spaces


def filter_character(string, character):
    return string.replace(character, "")


# def pred_sentence_to_string(original_string, original_sentence, pred_sentence):
#     spaces = add_spaces(original_string, original_sentence)
#
#     result = ""
#     for i in range(len(sentence)):
#         word = sentence[i]
#
#         filtered_word = filter_character(word, "$")
#         if len(filtered_word) != 0:
#             result += filtered_word
#
#             if spaces[i]:
#                 result += " "
#
#     return result


def pred_to_string(original_string, original_sentence, prediction):
    spaces = add_spaces(original_string, original_sentence)

    result = ""
    for i in range(len(original_sentence)):
        word = filter_character(original_sentence[i].copy(), "$")

        if prediction[i]:
            result += word

            if spaces[i]:
                result += " "

    return result
