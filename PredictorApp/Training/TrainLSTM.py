import datetime
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from DataProcessing.DataGrouper import group_tweetsdata, group_volumedata, group_bitcoindata
from Device import DEVICE
from FeatureExtraction import *
from Models.LSTM.LSTM import LSTM
import os

# region CONFIG
TIME_INTERVAL = "3H"


# endregion


def load_tweet_data(time_interval, files, renew=False):
    print("READING TWEETS")
    potential_path = "../Data/2018-Weighted/cache/" + "TWEETS" + "(" + time_interval + ")" + ".csv"

    if os.path.isfile(potential_path) and not renew:
        # Read CACHED FILE
        print(" - Cached File Found")
        df = pd.read_csv(potential_path)
        print(" - LOADED", potential_path)
        return df

    print(" - NO Cached File Found")

    root_path = "../Data/2018-Weighted/grouped/"
    tweets_data = pd.DataFrame()
    for file_path in files:
        print("     - reading", file_path)
        temp = pd.read_csv(root_path + file_path)
        tweets_data = tweets_data.append(temp)

    print(" - Grouping")
    grouped_df = group_tweetsdata(tweets_data, time_interval)
    print(" - Writing")
    grouped_df.to_csv(potential_path)
    print(" - Tweets Data Cached in", potential_path)
    return grouped_df


def load_volume_data(time_interval, start_date, end_date, renew=False):
    print("READING VOLUME")
    potential_path = "../Data/2018-Weighted/cache/" + "VOLUME" + "(" + time_interval + ")" + ".csv"

    if os.path.isfile(potential_path) and not renew:
        # Read CACHED FILE
        print(" - Cached File Found")
        df = pd.read_csv(potential_path)
        return df
    print(" - Cached File NOT Found")

    file_path = "../Data/2018tweets/2018(03-08--03-11).csv"
    print("     - reading", file_path)
    volume_data = pd.read_csv(file_path)

    print(" - Grouping")
    grouped_df = group_volumedata(volume_data, time_interval)

    print("     - Filtering Between", start_date.date(), "and", end_date.date())
    grouped_df = grouped_df.loc[(grouped_df['date'] >= start_date.replace(tzinfo=None))
                                & (grouped_df['date'] < end_date.replace(tzinfo=None))]

    print(" - Writing")
    grouped_df.to_csv(potential_path)
    print(" - Volume Data Cached in", potential_path)
    return grouped_df


def load_bitcoin_data(time_interval, start_date, end_date, renew=False):
    print("READING BITCOIN")
    potential_path = "../Data/2018-Weighted/cache/" + "BITCOIN" + "(" + time_interval + ")" + ".csv"

    if os.path.isfile(potential_path) and not renew:
        # Read CACHED FILE
        print(" - Cached File Found")
        bitcoin_df = pd.read_csv(potential_path)
        return bitcoin_df
    print(" - Cached File NOT Found")

    file_path = "../Data/bitcoin/Bitstamp_BTCUSD_2018_minute.csv"
    print("     - reading", file_path)
    bitcoin_data = pd.read_csv(file_path)

    print("     - Filtering Between", start_date.date(), "and", end_date.date())
    # bitcoin_data = bitcoin_data.sort_values(by='Date')
    bitcoin_data['Date'] = pd.to_datetime(bitcoin_data['Date'])
    bitcoin_data = bitcoin_data.loc[(bitcoin_data['Date'] >= start_date.replace(tzinfo=None))
                                    & (bitcoin_data['Date'] < end_date.replace(tzinfo=None))]

    print(" - Grouping")
    grouped_df = group_bitcoindata(bitcoin_data, time_interval)

    print(" - Writing")
    grouped_df.to_csv(potential_path)
    print(" - Bitcoin Data Cached in", potential_path)
    return grouped_df


if __name__ == '__main__':
    # region Load_Data
    files = [
        "03 2018(1Min).csv",
        "04 2018(1Min).csv",
        "05 2018(1Min).csv",
        "06 2018(1Min).csv",
        "07 2018(1Min).csv",
        "08 2018(1Min).csv",
        "09 2018(1Min).csv",
        "10 2018(1Min).csv",
        "11 2018(1Min).csv"
    ]

    start_date = datetime.datetime(2018, 3, 8, 0, 0, 0, 0, datetime.timezone.utc)
    end_date = datetime.datetime(2018, 11, 4, 0, 0, 0, 0, datetime.timezone.utc)

    tweet_df = load_tweet_data(TIME_INTERVAL, files)
    volume_df = load_volume_data(TIME_INTERVAL, start_date, end_date)
    bitcoin_df = load_bitcoin_data(TIME_INTERVAL, start_date, end_date)
    # endregion
