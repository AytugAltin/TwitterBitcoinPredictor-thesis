import numpy as np
import pandas as pd
from time import time

import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from twython import Twython, TwythonError
from Keys import *

import BotClassifier


def sentiment_scores(sentence):
    print("--", sentence, "--")
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)

    print("Overall sentiment dictionary is : ", sentiment_dict)
    print("sentence was rated as ", sentiment_dict['neg'] * 100, "% Negative")
    print("sentence was rated as ", sentiment_dict['neu'] * 100, "% Neutral")
    print("sentence was rated as ", sentiment_dict['pos'] * 100, "% Positive")

    print("Sentence Overall Rated As", end=" ")

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05:
        print("Positive")
    elif sentiment_dict['compound'] <= - 0.05:
        print("Negative")
    else:
        print("Neutral")


def sentiment_testing():
    sentences = [
        "This was a not good movie.",
    ]
    for sentence in sentences:
        sentiment_scores(sentence)
        print("")


if __name__ == '__main__':
    app_key = API_KEY
    app_secret = API_SECRET_KEY
    oauth_token = ACCESS_TOKEN
    oauth_token_secret = ACCESS_TOKEN_SECRET

    naughty_words = [" -RT"]
    good_words = ["bitcoin", "btc"]
    filter = " OR ".join(good_words)
    blacklist = " -".join(naughty_words)
    keywords = filter + blacklist

    twitter = Twython(app_key, app_secret, oauth_token, oauth_token_secret)
    search_results = twitter.search(q=keywords, count=100)
    print('tes')
