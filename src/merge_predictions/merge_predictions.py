from nltk.tokenize import WordPunctTokenizer


def get_pred_tweet(tweet, prediction):
    list_words = WordPunctTokenizer().tokenize(tweet)


def pred_tweet(tweets, predictions):
    nb_tweets = len(tweets)
    pred_tweets = []

    for idx_tweet in range(nb_tweets):
        pred_tweets.append(get_pred_tweet(tweets[idx_tweet], predictions[idx_tweet]))

    return pred_tweets

if __name__ == "__main__":
