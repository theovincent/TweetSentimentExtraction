from src.descriptors.tokenizer.modify_nltk.modified_tweet_tokenizer import TweetTokenizerReducedLen


class Tokenizer:
    def __init__(self, reduce_len=False, strip_handles=True):
        self.tokenizer = TweetTokenizerReducedLen(reduce_len=reduce_len, strip_handles=strip_handles)

    def tokenize(self, tweet, sentence_size):
        return self.tokenizer.tokenize(tweet)[: sentence_size]


if __name__ == "__main__":
    TWEET = "***** @ENPC While im stuck INSIDE in Elk Grove Village working all day   Someone should enjoy it!"
    SENTENCE_SIZE = 40

    TOKENIZER = Tokenizer()
    print(TOKENIZER.tokenize(TWEET, SENTENCE_SIZE))
