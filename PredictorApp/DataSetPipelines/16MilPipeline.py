import datetime

import pandas as pd
import os


def preprocessing_mix():
    filelist = []
    root_path = "../raw_data/mix/Raw/18/PART 2 complete"

    for roots, dirs, files in os.walk(root_path):
        for file in files:
            filelist.append(os.path.join(roots, file))

    tweets_data = pd.DataFrame()
    for file_path in filelist:
        temp = pd.read_csv(file_path, sep=";", error_bad_lines=False)

        tweets_data = tweets_data.append(temp)

        print("Loaded", file_path, " ", temp.shape)

    tweets_data.to_csv("raw_data/mix/Per Period/" + "2018_part1.csv")

    print("Shape ", tweets_data.shape)
    print("done")


def create_dataset_mix():
    start_date = datetime.datetime(2017, 5, 31, 0, 0, 0, 0, datetime.timezone.utc)
    end_date = datetime.datetime(2019, 6, 1, 23, 0, 0, 0, datetime.timezone.utc)

    volume_data = pd.read_csv("../raw_data/mix/Per Period/2018_part1.csv")

    volume_data = volume_data.sort_values(by='date')

    dataset = \
        DatasetBtcTweets(
            tweets_data=volume_data,
            volume_data=volume_data,
            bitcoin_data=volume_data,
            time_interval="60Min"
        )


def create_dataset_16m():
    start_date = datetime.datetime(2017, 1, 1, 0, 0, 0, 0, datetime.timezone.utc)
    end_date = datetime.datetime(2019, 1, 27, 23, 0, 0, 0, datetime.timezone.utc)

    tweets_data = pd.read_csv("../raw_data/TweetsBTC_16mil/filtered/17_en_filtered.csv", lineterminator='\n')
    tweets_data["date"] = tweets_data["timestamp"]
    tweets_data = tweets_data.sort_values(by='date')
    data = tweets_data
    tweets_data = pd.read_csv("../raw_data/TweetsBTC_16mil/filtered/18_en_filtered.csv", lineterminator='\n')
    tweets_data["date"] = tweets_data["timestamp"]
    tweets_data = tweets_data.sort_values(by='date')
    data = data.append(tweets_data)
    tweets_data = pd.read_csv("../raw_data/TweetsBTC_16mil/filtered/19_en_filtered.csv", lineterminator='\n')
    tweets_data["date"] = tweets_data["timestamp"]
    tweets_data = tweets_data.sort_values(by='date')
    tweets_data = data.append(tweets_data)

    tweets_data['date'] = pd.to_datetime(tweets_data['date'])

    tweets_data = tweets_data.loc[
        (tweets_data['date'] >= start_date) &
        (tweets_data['date'] <= end_date)
        ]

    volume_data = pd.read_csv("../raw_data/TweetsBTC_16mil/peryear/17.csv", lineterminator='\n')
    volume_data["date"] = volume_data["timestamp"]
    volume_data = volume_data.sort_values(by='date')
    data = volume_data
    volume_data = pd.read_csv("../raw_data/TweetsBTC_16mil/peryear/18.csv", lineterminator='\n')
    volume_data["date"] = volume_data["timestamp"]
    volume_data = volume_data.sort_values(by='date')
    data = data.append(volume_data)
    volume_data = pd.read_csv("../raw_data/TweetsBTC_16mil/peryear/19.csv", lineterminator='\n')
    volume_data["date"] = volume_data["timestamp"]
    volume_data = volume_data.sort_values(by='date')
    volume_data = data.append(volume_data)

    volume_data['date'] = pd.to_datetime(volume_data['date'])

    volume_data = volume_data.loc[
        (volume_data['date'] >= start_date) &
        (volume_data['date'] <= end_date)
        ]

    bitcoin_data = pd.read_csv("../raw_data/bitcoin/Bitstamp_BTCUSD_2017_minute.csv")
    bitcoin_data = bitcoin_data.sort_values(by='Date')
    data = bitcoin_data
    bitcoin_data = pd.read_csv("../raw_data/bitcoin/Bitstamp_BTCUSD_2018_minute.csv")
    bitcoin_data = bitcoin_data.sort_values(by='Date')
    data = data.append(bitcoin_data)
    bitcoin_data = pd.read_csv("../raw_data/bitcoin/Bitstamp_BTCUSD_2019_minute.csv")
    bitcoin_data = bitcoin_data.sort_values(by='Date')
    bitcoin_data = data.append(bitcoin_data)

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
    return dataset


if __name__ == '__main__':
    preprocessing_mix()
