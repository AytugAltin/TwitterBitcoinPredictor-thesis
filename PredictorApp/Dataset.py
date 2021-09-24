import torch
from torch.utils.data import Dataset
from torchvision import datasets
import pandas as pd
import numpy as np
import ast


def meanlist(listoflists):
    divider = len(listoflists)
    try:
        summed = listoflists.pop()
        while len(listoflists) > 0:
            summed = [sum(item) for item in zip(summed, listoflists.pop())]
        mean = []
        for number in summed:
            mean.append(number / divider)
        return mean
    except:
        return []


class DatasetBtcTweets(Dataset):
    def __init__(self, tweets_data, bitcoin_data, volume_data, time_interval, bot_filtering=True,
                 transform=None):
        self.data = pd.DataFrame()
        self.time_interval = time_interval
        self.tweets_volume = self.create_volumedata(volume_data)
        print("Loaded tweets volume")
        self.bitoin_data = self.create_bitcoindata(bitcoin_data)
        print("Loaded bitcoin data")
        self.tweets_data = self.create_tweetsdata(tweets_data, bot_filtering)
        print("Loaded tweets data")


    def __len__(self):
        return self.data

    def __getitem__(self, index):
        image = self.data.iloc[index, 1:]
        label = self.data.iloc[index, 0]

        return image, label

    def create_tweetsdata(self, tweets, bot_filtering):
        data = tweets

        data = data.sort_values(by='date')
        data = data[["date", "bert", "sent_neg", "sent_neu", "sent_pos", "bot"]]

        data = data[data.bot != bot_filtering][["date", "bert", "sent_neg", "sent_neu", "sent_pos"]]

        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index("date")
        data.index = pd.to_datetime(data.index)
        aggregations = {
            'sent_neg': 'mean',
            'sent_neu': 'mean',
            'sent_pos': 'mean',
            "bert": lambda x: meanlist([ast.literal_eval(y) for y in x.values])
        }
        grouped_data = data.groupby(pd.Grouper(freq=self.time_interval)).agg(aggregations)

        return grouped_data

    def create_volumedata(self, volume_data):
        data = volume_data
        data = data.sort_values(by='date')
        data = data[["date"]]

        data = data.set_index("date")
        data.index = pd.to_datetime(data.index)

        grouped_data = data.groupby(pd.Grouper(freq=self.time_interval)).size().reset_index(name='tweet_vol')
        return grouped_data

    def create_bitcoindata(self, bitcoin_data):
        data = bitcoin_data
        data = data.sort_values(by='Date')
        data = data[["Date", "Open", "High", "Low", "Close", "Volume", "Volume USD"]]
        data = data.set_index("Date")
        data.index = pd.to_datetime(data.index)

        aggregations = {
            'Open': lambda x: x.iloc[0],
            'High': 'max',
            'Low': 'min',
            "Close": lambda x: x.iloc[-1],
            "Volume": 'mean'
        }

        grouped_data = data.groupby(pd.Grouper(freq=self.time_interval)).agg(aggregations)
        return grouped_data

    def print_status(self):
        self.tweets_data["bert"].count([])
