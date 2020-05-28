import numpy as np


def one_hot_representation(sentence, sentence_size, word_size, fill_with_ones=False):
    size = 90 - 32 + 1  # number of different signs we encode

    inputs = sentence
    outputs = np.zeros((sentence_size, word_size, size))

    for i, word in enumerate(inputs[:sentence_size]):
        word = word[:word_size].upper()
        output = []

        for character in word:
            number = ord(character) - 32
            if number < 0 or number > size:
                number = -1
            output.append(number)

        output = np.array(output)

        nbr_signs = output.size
        one_hot = np.zeros((nbr_signs, size + 1))
        one_hot[np.arange(nbr_signs), output] = 1

        outputs[i, :nbr_signs, :] = one_hot[:, :-1]
        if fill_with_ones:
            outputs[i, nbr_signs:, :] = 1

    return outputs


def descriptor_one_hot(x_sentence, sentence_size, word_size,
                       feelings=None, feeling_weight=1, fill_with_ones=False):
    x_one_hot = []
    for i, sentence in enumerate(x_sentence):
        sentence_one_hot = one_hot_representation(sentence, sentence_size, word_size, fill_with_ones)
        sentence_one_hot = sentence_one_hot.flatten()

        if feelings is not None:
            sentence_one_hot = np.concatenate((sentence_one_hot, [feelings[i] * feeling_weight]))

        x_one_hot.append(sentence_one_hot)

    return np.array(x_one_hot)


if __name__ == '__main__':
    EX = ["i", "feel", "really", "bored", "."]
    EX_LABEL = ["bored", "."]

    EX_ONE_HOT = one_hot_representation(EX, 6, 10)

    print(EX_ONE_HOT.shape)
    print(EX_ONE_HOT[2])

    EX2 = [["i", "feel", "really", "bored", "."],
           ["we", "are", "champions"]
           ]

    FEELINGS = [1, -1]
    EX2_ONE_HOT = descriptor_one_hot(EX2, 6, 10, FEELINGS, 10)

    print(EX2_ONE_HOT.shape)

    # print the feeling
    print(EX2_ONE_HOT[0, -1])
