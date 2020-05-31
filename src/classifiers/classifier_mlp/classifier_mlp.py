import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.activations import sigmoid
from tensorflow.python.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    MaxPooling2D,
    ReLU
)


class ClassifierDense(Model):
    def __init__(self, word_size, sentence_size):
        super(ClassifierDense, self).__init__()
        # Parameters
        self.word_size = word_size
        self.sentence_size = sentence_size

        # Functions
        self.dense1 = Dense(word_size * sentence_size)
        self.activation1 = ReLU()

        self.dense2 = Dense(sentence_size)

        self.sigmoid = sigmoid

    def call(self, inputs):
        # Layer 1
        # WORD_SIZE * SENTENCE_SIZE
        x = self.dense1(inputs)
        x = self.activation1(x)
        # WORD_SIZE * SENTENCE_SIZE

        # Layer 2
        # WORD_SIZE * SENTENCE_SIZE
        x = self.dense2(x)
        # SENTENCE_SIZE

        x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    # Parameters
    WORD_SIZE = 5
    SENTENCE_SIZE = 3

    # Define the classifier
    CLASSIFIFIER = ClassifierDense(WORD_SIZE, SENTENCE_SIZE)

    # Data
    DATA_SCALAR = np.random.random((10, WORD_SIZE * SENTENCE_SIZE))
    DATA_SCALAR = np.array(DATA_SCALAR, dtype=np.float32)
    # print(DATA_SCALAR)

    # Apply the classifier
    CLASSIFIFIER.trainable = False
    PREDICTION = CLASSIFIFIER(DATA_SCALAR)

    print(PREDICTION)
