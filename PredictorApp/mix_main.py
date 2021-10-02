import datetime

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from BotClassifier import BotClassifier
from Dataset import DatasetBtcTweets
from Keys import BEARER_TOKEN
from TweetScraper import TweetScraper
import os
from os.path import isfile, join
import pickle

def preprocessing_mix():
    filelist = []
    root_path = "data/mix/Raw/18/PART 2 complete/"

    for roots, dirs, files in os.walk(root_path):
        for file in files:
            filelist.append(os.path.join(roots, file))

    tweets_data = pd.DataFrame()
    for file_path in filelist:
        try:
            temp = pd.read_csv(file_path, sep=";")
        except:
            temp = pd.read_csv(file_path, lineterminator="\n")

        tweets_data = tweets_data.append(temp)

        print("Loaded", file_path, " ", temp.shape)

    tweets_data.to_csv("data/mix/Per Period/" + "2018(03-08--03-11).csv")

    print("Shape ", tweets_data.shape)
    print("done")


def create_dataset_mix():
    start_date = datetime.datetime(2018, 3, 8, 0, 0, 0, 0, datetime.timezone.utc)
    end_date = datetime.datetime(2019, 11, 4, 0, 0, 0, 0, datetime.timezone.utc)

    tweets_data = pd.read_csv("data/mix/Per Period/2018(03-08--03-11)_en_filtered.csv"
                              , lineterminator='\n')

    volume_data = pd.read_csv("data/mix/Per Period/2018(03-08--03-11).csv"
                              , lineterminator='\n')

    bitcoin_data = pd.read_csv("data/bitcoin/Bitstamp_BTCUSD_2018_minute.csv")
    bitcoin_data = bitcoin_data.sort_values(by='Date')
    bitcoin_data['Date'] = pd.to_datetime(bitcoin_data['Date'])
    bitcoin_data = bitcoin_data.loc[(bitcoin_data['Date'] >= start_date.replace(tzinfo=None))
                                    & (bitcoin_data['Date'] < end_date.replace(tzinfo=None))]

    dataset = \
        DatasetBtcTweets(
            tweets_data=tweets_data,
            volume_data=volume_data,
            bitcoin_data=bitcoin_data,
            time_interval="60Min"
        )
    with open('filename.pickle', 'wb') as handle:
        pickle.dump(dataset, handle)


if __name__ == '__main__':
    create_dataset_mix()
