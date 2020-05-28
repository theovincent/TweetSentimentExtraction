from nltk.tokenize import TweetTokenizer


class Tokenizer:
    def __init__(self, strip_handles=True):
        self.tokenizer = TweetTokenizer(strip_handles=strip_handles)

    def tokenize(self, tweet, sentence_size):
        # Decomposition
        word_decompostion = self.tokenizer.tokenize(tweet)[: sentence_size]

        return word_decompostion


if __name__ == "__main__":
    TWEET = "While im stuck INSIDE in Elk Grove Village working all day   Someone should enjoy it!"
    SENTENCE_SIZE = 20

    TOKENIZER = Tokenizer()
    print(TOKENIZER.tokenize(TWEET, SENTENCE_SIZE))
