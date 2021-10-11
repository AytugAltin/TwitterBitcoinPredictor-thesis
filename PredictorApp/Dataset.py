import torch
from torch.utils.data import Dataset
from torchvision import datasets
import pandas as pd
import numpy as np
import ast


def meanlist(listoflists):
    divider = len(listoflists)
    try:
        summed = listoflists[0]
        for index in range(1, divider):
            summed = [sum(item) for item in zip(summed, listoflists[index])]
        return summed
    except:
        return []


def convert_to_datetime(data_frame, key):
    try:
        data_frame[key] = pd.to_datetime(data_frame[key], format='%d/%m/%Y %H:%M:%S', utc=True)
    except:
        try:
            data_frame[key] = pd.to_datetime(data_frame[key], format='%d/%m/%Y %H:%M', utc=True)
        except:
            data_frame[key] = pd.to_datetime(data_frame[key], format='%Y-%m-%d %H:%M:%S', utc=True)

    return data_frame


def multiply_row_by_count(row, multiplyer):
    row["sent_neg"] = row["sent_neg"] * multiplyer
    row["sent_neu"] = row["sent_neu"] * multiplyer
    row["sent_pos"] = row["sent_pos"] * multiplyer

    try:
        row["bert"] = list(map(float, list(row["bert"].strip('[]').split(','))))
    except:
        pass

    row["bert"] = [element * multiplyer for element in row["bert"]]

    return row


class DatasetBtcTweets(Dataset):
    def __init__(self, tweets_data, bitcoin_data, volume_data, time_interval, bot_filtering=True,
                 transform=None):
        self.time_interval = time_interval
        self.tweets_data = self.create_tweetsdata(tweets_data, bot_filtering)
        print("Loaded tweets data")
        self.tweets_volume = self.create_volumedata(volume_data)
        print("Loaded tweets volume")
        self.bitoin_data = self.create_bitcoindata(bitcoin_data)
        print("Loaded bitcoin data")

    def __len__(self):
        return self.data

    def __getitem__(self, index):
        tweet_data = self.tweets_data.iloc[index, :]
        volume = self.tweets_volume.iloc[index, :]
        bitcoin_data = self.bitoin_data.iloc[index, :]

        return tweet_data, volume, bitcoin_data

    def create_tweetsdata(self, tweets, bot_filtering):
        data = tweets
        data = data.sort_values(by='date')

        data = convert_to_datetime(data, "date")
        data['date'] = data["date"].dt.tz_localize(None)
        data = data.set_index("date")

        data = data.apply(lambda row: multiply_row_by_count(row, row["count"]), axis=1)

        aggregations = {
            'sent_neg': 'sum',
            'sent_neu': 'sum',
            'sent_pos': 'sum',
            "bert": lambda x: meanlist(x.values),
            "count": 'sum'
        }
        grouped_data = data.groupby(pd.Grouper(freq=self.time_interval)).agg(aggregations)

        grouped_data = grouped_data.apply(lambda row: multiply_row_by_count(row, 1 / row["count"]), axis=1)

        return grouped_data

    def create_volumedata(self, volume_data):
        data = volume_data
        data = data.sort_values(by='date')
        data = data[["date"]]

        data = convert_to_datetime(data, "date")
        data['date'] = data["date"].dt.tz_localize(None)
        data = data.set_index("date")


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
