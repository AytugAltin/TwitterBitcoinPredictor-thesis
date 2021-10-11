import time

import pandas as pd
from DataProcessing.BERTTweet import BERTTweet


class BERTCalculator:

    def __init__(self):
        self.tweet_bert = BERTTweet()

    def pre_processing(self, tweet):
        tweet = tweet.replace('\n', '')
        tweet = tweet.replace('\t', '')
        tweet = tweet.replace('\r', '')
        return tweet

    def pipeline(self, index, row, data_set):
        tweet = row["text"]

        tweet = self.pre_processing(tweet)

        vec = self.tweet_bert.tweet_to_vec_string(tweet)
        data_set._set_value(index, 'bert', vec)


def vectorize_tweets(data):
    print("VectorizeBert started")
    mechanism = BERTCalculator()

    data["bert"] = [[]] * len(data)
    t0 = time.time()

    for index, row in data.iterrows():

        try:
            mechanism.pipeline(index, row, data)

        except IndexError as err:
            print(err)

        except Exception as err:
            print(row["text"])
            raise err

        if index % 10000 == 0:
            print("index", index, " time: ", (time.time() - t0))

    print("VectorizeBert done")

    return data
