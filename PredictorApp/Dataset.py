import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


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


def create_tweetsdata(self, tweets, bot_filtering=True):
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
    grouped_data = grouped_data.set_index("date")
    return grouped_data


def print_status(self):
    self.tweets_data["bert"].count([])


class DataSetReader(Dataset):
    def __init__(self, csv_file):
        self.data = convert_to_datetime(self.data, "date")
        self.data['date'] = self.data["date"].dt.tz_localize(None)
        self.data = self.data.set_index("date")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx, 1:]
        label = self.data.iloc[idx, 1:]

        return sample, label


