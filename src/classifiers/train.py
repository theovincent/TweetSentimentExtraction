import numpy as np
from tensorflow.python.keras.losses import binary_crossentropy
from classifiers.classifier_mlp.classifier_mlp import ClassifierConv


# Parameters
WORD_SIZE = 5
SENTENCE_SIZE = 5


# The data
TRAIN_SCALAR = np.random.random((10, WORD_SIZE * SENTENCE_SIZE))
TRAIN_IMPORTANT_WORDS = np.random.randint(0, 2, (10, SENTENCE_SIZE))

VALID_SCALAR = np.random.random((5, WORD_SIZE * SENTENCE_SIZE))
VALID_IMPORTANT_WORDS = np.random.randint(0, 2, (5, SENTENCE_SIZE))


# Transform for classifier
TRAIN_SCALAR_CLASS = np.reshape(TRAIN_SCALAR, (10, SENTENCE_SIZE, WORD_SIZE))
VALID_SCALAR_CLASS = np.reshape(VALID_SCALAR, (5, SENTENCE_SIZE, WORD_SIZE))
VALID_DATA = (VALID_SCALAR_CLASS, VALID_IMPORTANT_WORDS)


# The classifier
CLASSIFIER = ClassifierConv(WORD_SIZE, SENTENCE_SIZE)

# Compile the classifier
CLASSIFIER.compile(optimizer='adam', loss=binary_crossentropy, metrics=['accuracy'])

CLASSIFIER.fit(TRAIN_SCALAR_CLASS, TRAIN_IMPORTANT_WORDS, batch_size=2, epochs=10, validation_data=VALID_DATA, shuffle=True,
               class_weight=None)
