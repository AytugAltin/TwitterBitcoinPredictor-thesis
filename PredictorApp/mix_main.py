import datetime
import pandas as pd

from CalculateBERT import vectorize_tweets
from ClassifyTweets import classify
from Dataset import DatasetBtcTweets
# from LanguageFiltering import filter_language
import os
import pickle


def preprocessing_mix():
    folders = ["03 2018", "04 2018", "05 2018", "06 2018", "07 2018",
               "08 2018", "09 2018", "10 2018", "11 2018"]
    for folder in folders:
        filelist = []
        root_path = "raw_data/mix/Raw/18/PART 2 complete/" + folder

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

        tweets_data.to_csv("Data/2018tweets/unprocessed/" + folder + ".csv")

        print("Shape ", tweets_data.shape)

    print("done")


def language_filtering():
    root_path = "Data/2018tweets/unprocessed/"

    files = ["03 2018", "04 2018", "05 2018", "06 2018", "07 2018",
             "08 2018", "09 2018", "10 2018", "11 2018"]

    for file in files:
        fields = ["date", "text"]
        try:
            data = pd.read_csv(root_path + file + ".csv", sep=";", usecols=fields)
        except:
            data = pd.read_csv(root_path + file + ".csv", lineterminator="\n", usecols=fields)

        print("Loaded", file, " ", data.shape)

        # en_df, data = filter_language(data)
        data.to_csv(path_or_buf="Data/2018tweets/languagefiltering/" + file + "_lang.csv")
        en_df.to_csv(path_or_buf="Data/2018tweets/languagefiltering/" + file + "_en.csv")


def classify_tweets():
    print("classify_tweets")
    root_path = "Data/2018tweets/languagefiltering/"

    files = ["03 2018_en", "04 2018_en", "05 2018_en", "06 2018_en", "07 2018_en",
             "08 2018_en", "09 2018_en", "10 2018_en", "11 2018_en"]

    for file in files:
        try:
            data = pd.read_csv(root_path + file + ".csv", sep=";")
        except:
            data = pd.read_csv(root_path + file + ".csv", lineterminator="\n")
        print("Loaded", file, " ", data.shape)

        data = classify(data)
        data.to_csv(path_or_buf="Data/2018tweets/classed/" + file + ".csv")


def vec_tweets():
    print("classify_tweets")
    root_path = "Data/2018tweets/classed/"

    files = ["03 2018_en", "04 2018_en", "05 2018_en", "06 2018_en", "07 2018_en",
             "08 2018_en", "09 2018_en", "10 2018_en", "11 2018_en"]
    files.reverse()

    for file in files:
        try:
            data = pd.read_csv(root_path + file + ".csv", sep=";")
        except:
            data = pd.read_csv(root_path + file + ".csv", lineterminator="\n")

        print("Loaded", file, " ", data.shape)

        data = vectorize_tweets(data)
        data.to_csv(path_or_buf="Data/2018tweets/bert/" + file + ".csv")


def create_dataset_mix(time_interval="60Min"):
    print("create_dataset_mix()")
    start_date = datetime.datetime(2018, 3, 8, 0, 0, 0, 0, datetime.timezone.utc)
    end_date = datetime.datetime(2018, 11, 4, 0, 0, 0, 0, datetime.timezone.utc)

    print("reading bitcoin_data")
    bitcoin_data = pd.read_csv("raw_data/bitcoin/Bitstamp_BTCUSD_2018_minute.csv")
    bitcoin_data = bitcoin_data.sort_values(by='Date')
    bitcoin_data['Date'] = pd.to_datetime(bitcoin_data['Date'])
    bitcoin_data = bitcoin_data.loc[(bitcoin_data['Date'] >= start_date.replace(tzinfo=None))
                                    & (bitcoin_data['Date'] < end_date.replace(tzinfo=None))]

    print("reading tweets_data")
    tweets_data = pd.read_csv("raw_data/mix/Per Period/2018(03-08--03-11)_en_filtered.csv"
                              , lineterminator='\n')

    print("reading volume_data")
    volume_data = pd.read_csv("raw_data/mix/Per Period/2018(03-08--03-11).csv"
                              , lineterminator='\n')

    print("creating DatasetBtcTweets")
    dataset = \
        DatasetBtcTweets(
            tweets_data=tweets_data,
            volume_data=volume_data,
            bitcoin_data=bitcoin_data,
            time_interval="60Min"
        )

    print("Saving DatasetBtcTweets")
    with open('data/DatasetBtcTweets/mix(' + time_interval + ').pickle', 'wb') as handle:
        pickle.dump(dataset, handle)

    print("create_dataset_mix Done")


if __name__ == '__main__':
    # preprocessing_mix()
    # language_filtering()
    # classify_tweets()
    vec_tweets()
    # create_dataset_mix(time_interval="60Min")
