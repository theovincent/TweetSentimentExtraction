from nltk.tokenize import TweetTokenizer


class Tokenizer:
    def __init__(self, strip_handles=True):
        self.tokenizer = TweetTokenizer(strip_handles=strip_handles)

    def tokenize(self, tweet):
        return self.tokenizer.tokenize(tweet)


if __name__ == "__main__":
    TWEET = "While im stuck INSIDE in Elk Grove Village working all day   Someone should enjoy it!"
    TOKENIZER = Tokenizer()
    print(TOKENIZER.tokenize(TWEET))
