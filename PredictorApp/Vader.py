from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class Vader:

    def __init__(self):
        self.SIA = SentimentIntensityAnalyzer()

    def get_score(self, tweet):
        sentiment_dict = self.SIA.polarity_scores(tweet)
        return sentiment_dict
