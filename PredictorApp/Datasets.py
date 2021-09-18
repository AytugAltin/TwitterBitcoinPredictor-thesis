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
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path, lineterminator='\n')

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data.iloc[index, 1:]
        label = self.data.iloc[index, 0]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DatasetCreator:
    def __init__(self, tweets_path, bitcoin_path, time_interval, volume_path, bot_filtering=True):
        self.data = pd.DataFrame()
        self.time_interval = time_interval
        self.bitoin_data = self.create_bitcoindata(bitcoin_path)
        self.tweets_data = self.create_tweetsdata(tweets_path, bot_filtering)
        print("Loaded tweets data")
        self.tweets_volume = self.create_volumedata(volume_path)

    def create_tweetsdata(self, tweets_path, bot_filtering):
        data = pd.read_csv(tweets_path, lineterminator='\n')

        data = data.sort_values(by='timestamp')
        data = data[["timestamp", "bert", "sent_neg", "sent_neu", "sent_pos", "bot"]]

        data = data[data.bot != bot_filtering][["timestamp", "bert", "sent_neg", "sent_neu", "sent_pos"]]

        data['date'] = pd.to_datetime(data['timestamp'])
        data = data.set_index("timestamp")
        data.index = pd.to_datetime(data.index)
        aggregations = {
            'sent_neg': 'mean',
            'sent_neu': 'mean',
            'sent_pos': 'mean',
            "bert": lambda x: meanlist([ast.literal_eval(y) for y in x.values])
        }
        data = data.groupby(pd.Grouper(freq=self.time_interval)).agg(aggregations)

        return data

    def create_volumedata(self, volume_path):
        data = pd.read_csv(volume_path, lineterminator='\n')
        data = data.sort_values(by='timestamp')
        data = data[["timestamp"]]

        data = data.set_index("timestamp")
        data.index = pd.to_datetime(data.index)

        data = data.groupby(pd.Grouper(freq=self.time_interval)).size().reset_index(name='tweet_vol')
        return data

    def create_bitcoindata(self, bitcoin_path):
        data = pd.read_csv(bitcoin_path)
        data = data.sort_values(by='Date')
        data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
        data = data.set_index("Date")
        data.index = pd.to_datetime(data.index)

        aggregations = {
            'Open': lambda x: x.iloc[0],
            'High': 'max',
            'Low': 'min',
            "Close": lambda x: x.iloc[-1],
            "Volume": 'mean'
        }

        data = data.groupby(pd.Grouper(freq=self.time_interval)).agg(aggregations)

        return data
