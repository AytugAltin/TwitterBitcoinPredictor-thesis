import nltk

# from paper "Predicting Bitcoin price fluctuation with Twitter sentiment analysis"
HASHTAGS = ["mpgvip", "freebitcoin", "livescore", "makeyourownlane", "footballcoin"]
WORDS = ["entertaining", "subscribe"]
BIGRAMS = [("free", "bitcoin"), ("current", "price"), ("bitcoin", "price"), ("earn", "bitcoin")]
TRIGRAMS = [("start", "trading", "bitcoin")]


class BotClassifier:

    def __init__(self):
        print()

    def tweet_is_bot(self, tweet):
        tweet = tweet.lower()

        if self.hashtags(tweet):
            return True

        if self.words(tweet):
            return True

        if self.bigrams(tweet):
            return True

        if self.trigrams(tweet):
            return True

        return False

    def hashtags(self, tweet):
        for hashtag in HASHTAGS:
            if hashtag in tweet:
                return True
        return False

    def words(self, tweet):
        for word in WORDS:
            if word in tweet:
                return True
        return False

    def bigrams(self, tweet):
        nltk_tokens = nltk.word_tokenize(tweet)
        tweet_bigrams = list(nltk.bigrams(nltk_tokens))
        for bot_bigram in BIGRAMS:
            if bot_bigram in tweet_bigrams:
                return True
        return False

    def trigrams(self, tweet):
        nltk_tokens = nltk.word_tokenize(tweet)
        tweet_trigrams = list(nltk.trigrams(nltk_tokens))
        for bot_trigram in TRIGRAMS:
            if bot_trigram in tweet_trigrams:
                return True
        return False
