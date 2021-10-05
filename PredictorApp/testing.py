import numpy as np
import pandas as pd
from time import time
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from Keys import *
from BERTTweet import BERTTweet
import BotClassifier
import torch
from transformers import AutoModel, AutoTokenizer

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
    strings = [
        "added video playlist bitcoin march st still possible",
        # [0, 8, 792, 11, 202, 9, 11, 5, 1171, 10, 6635, 144, 55026, 24, 1282, 14350, 1085, 12988, 21, 2]
        "stormx allows travelers purchase first class airline tickets cryptocurrency bitcoin",
        # [0, 34140, 777, 56155, 54101, 9, 14719, 847, 3784, 39937, 6025, 30, 40021, 15488, 67, 10, 2]

        "Better crypto buy?? $ETH $DASH $DGD $XMR $LTC $NEO $BTG $ETC $QTUM $LUN $GVT $OMG $NANO $HSR $BNB $EOS $WAVES $VEN $SALT $ICX $ARK $STEEM $BQX $BCPT $WINGS $APPC $GTO $BLZ $XLM $ADA $RPX $TRX $XVG $AION $VEN #Binance $BTCUSD $ICX $ADA $XMR #BITCOIN $TRX $VEN $NEM $SRN $GVT",
        # [0, 178, 149, 18746, 10, 28, 2],
        ]
    tweetbert = BERTTweet()

    for string in strings:
        print(string)
        print(tweetbert.tweet_to_vec(string))