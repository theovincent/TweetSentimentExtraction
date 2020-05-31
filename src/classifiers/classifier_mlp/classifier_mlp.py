import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.activations import sigmoid
from tensorflow.python.keras.layers import (
    Dense,
    Flatten,
    BatchNormalization,
    Conv1D,
    ReLU
)


class ClassifierDense(Model):
    def __init__(self, word_size, sentence_size):
        super(ClassifierDense, self).__init__()
        # Parameters
        self.word_size = word_size
        self.sentence_size = sentence_size

        # Functions
        self.dense1 = Dense(sentence_size)
        self.activation1 = ReLU()

        self.dense2 = Dense(sentence_size)

        self.sigmoid = sigmoid

    def call(self, inputs):
        # Layer 1
        # WORD_SIZE * SENTENCE_SIZE
        x = self.dense1(inputs)
        x = self.activation1(x)
        # SENTENCE_SIZE

        # Layer 2
        # SENTENCE_SIZE
        x = self.dense2(x)
        # SENTENCE_SIZE

        x = self.sigmoid(x)

        return x


class ClassifierConv(Model):
    def __init__(self, word_size, sentence_size):
        super(ClassifierConv, self).__init__()
        # Parameters
        self.word_size = word_size
        self.sentence_size = sentence_size

        # Functions
        self.conv1d = Conv1D(word_size // 4, 5, strides=1, padding='same', dilation_rate=2)
        self.batch_norm = BatchNormalization()
        self.activation1 = ReLU()

        self.flatten = Flatten()

        self.dense1 = Dense(sentence_size)
        self.sigmoid = sigmoid

    def call(self, inputs):
        # Layer 1
        # SENTENCE_SIZE x WORD_SIZE
        x = self.conv1d(inputs)
        x = self.batch_norm(x)
        x = self.activation1(x)
        # SENTENCE_SIZE x WORD_SIZE // 4

        x = self.flatten(x)
        # SENTENCE_SIZE * WORD_SIZE // 4

        # Layer 2
        # SENTENCE_SIZE * WORD_SIZE // 4
        x = self.dense1(x)
        x = self.sigmoid(x)
        # SENTENCE_SIZE

        return x


if __name__ == "__main__":
    # Parameters
    WORD_SIZE = 50
    SENTENCE_SIZE = 50

    # Define the classifier
    CLASSIFIFIER = ClassifierDense(WORD_SIZE, SENTENCE_SIZE)

    # Data
    DATA_SCALAR = np.random.random((10, WORD_SIZE * SENTENCE_SIZE))
    DATA_SCALAR = np.array(DATA_SCALAR, dtype=np.float32)
    # DATA_SCALAR_CLASS = np.reshape(DATA_SCALAR, (10, SENTENCE_SIZE, WORD_SIZE))
    # print(DATA_SCALAR_CLASS)

    # Apply the classifier
    PREDICTION = CLASSIFIFIER(DATA_SCALAR)
    CLASSIFIFIER.summary()

    print(PREDICTION)
