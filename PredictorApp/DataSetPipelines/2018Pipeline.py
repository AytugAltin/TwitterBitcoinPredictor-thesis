import datetime
import pandas as pd
import time
from DataProcessing.CalculateBERT import vectorize_tweets
from DataProcessing.ClassifyTweets import classify
from DataProcessing.TweetInfoRetriver import add_info_to_tweets

# from LanguageFiltering import filter_language
import os
import pickle
from DataProcessing.DataGrouper import *


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


def fill_weights():
    print("fill_weights")
    root_path = "../Data/2018tweets/unprocessed/"

    files = [
        "03 2018",
        "04 2018",
        "05 2018",
        "06 2018",
        # "07 2018",
        # "08 2018",
        # "09 2018",
        # "10 2018",
        # "11 2018"
    ]

    chunksize = 10000
    for file in files:
        print(file)
        chunk_list = []
        try:
            df_chunk = pd.read_csv(root_path + file + ".csv", lineterminator="\n", chunksize=chunksize)
        except:
            df_chunk = pd.read_csv(root_path + file + ".csv", sep=";", chunksize=chunksize)

        t0 = time.time()
        for chunk in df_chunk:
            print("Weighted", chunk.shape, " time: ", (time.time() - t0))
            finshed_chunk = add_info_to_tweets(chunk)
            chunk_list.append(finshed_chunk)
            finshed_chunk = None

        df_concat = pd.concat(chunk_list)
        df_concat.to_csv("Data/2018tweets/weighted/" + file + ".csv")


def language_filtering():
    root_path = "../Data/2018tweets/unprocessed/"

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
        # en_df.to_csv(path_or_buf="Data/2018tweets/languagefiltering/" + file + "_en.csv")


def classify_tweets():
    print("classify_tweets")
    root_path = "../Data/2018tweets/languagefiltering/"

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
    print("vec_tweets")
    root_path = "../Data/2018tweets/classed/"

    files = [
        "03 2018_en",
        "04 2018_en",
        # "05 2018_en",
        # "06 2018_en",
        # "07 2018_en","08 2018_en",
        # "09 2018_en",
        # "10 2018_en", "11 2018_en"
    ]

    for file in files:
        try:
            df_chunk = pd.read_csv(root_path + file + ".csv", lineterminator="\n", chunksize=100000)
        except:
            df_chunk = pd.read_csv(root_path + file + ".csv", sep=";", chunksize=100000)

        t0 = time.time()
        index = 0

        for chunk in df_chunk:
            print("vectorize", chunk.shape, " time: ", (time.time() - t0))
            data = vectorize_tweets(chunk)
            data.to_csv(path_or_buf="Data/2018tweets/bert/" + file + "_" + str(index) + ".csv")
            data = None
            print("Written", index, file)
            index = index + 1


def pre_group_data(time_interval):
    files = ["03 2018_en", "04 2018_en"]

    for file in files:
        print("pre group data " + file)

        df_chunk = pd.read_csv("Data/2018tweets/bert/" + file + ".csv"
                               , lineterminator='\n'
                               , chunksize=100000
                               )
        chunk_list = []
        t0 = time.time()
        for chunk in df_chunk:
            chunk = chunk.rename(columns={"bert\r": "bert"})
            chunk = chunk.rename(columns={"bert\r\r": "bert"})
            chunk_grouped = group_tweetsdata(chunk, time_interval)
            chunk_list.append(chunk_grouped)
            print("shape", chunk.shape, " time: ", (time.time() - t0))

        df_concat = pd.concat(chunk_list)
        df_concat.to_csv("Data/2018tweets/grouped/" + file + time_interval + ".csv")


def merge_files(endpath):
    root_path = "../Data/2018tweets/bert/"
    files = []

    t0 = time.time()
    chunk_list = []
    for file in files:
        print(file)
        try:
            data = pd.read_csv(root_path + file + ".csv", sep=";")
        except:
            data = pd.read_csv(root_path + file + ".csv", lineterminator="\n")

        chunk_list.append(data)
        print("shape", data.shape, " time: ", (time.time() - t0))

    df_concat = pd.concat(chunk_list)
    df_concat.to_csv(endpath)


def create_dataset_mix(time_interval="60Min"):
    print("create_dataset_mix()")
    start_date = datetime.datetime(2018, 3, 8, 0, 0, 0, 0, datetime.timezone.utc)
    end_date = datetime.datetime(2018, 11, 4, 0, 0, 0, 0, datetime.timezone.utc)

    print("reading tweets_data")
    filelist = [
        "03 2018_en1Min.csv",
        "04 2018_en1Min.csv",
        "05 2018_en1Min.csv",
        "06 2018_en1Min.csv",
        "07 2018_en1Min.csv",
        "08 2018_en1Min.csv",
        "09 2018_en1Min.csv",
        "10 2018_en1Min.csv",
        "11 2018_en1Min.csv"
    ]
    root_path = "../Data/2018tweets/grouped/1Min/"

    tweets_data = pd.DataFrame()
    for file_path in filelist:
        try:
            temp = pd.read_csv(root_path + file_path, lineterminator="\n")
        except:
            temp = pd.read_csv(root_path + file_path, sep=";")

        tweets_data = tweets_data.append(temp)

    tweets_data = tweets_data.dropna()
    new_columns = [i.strip() for i in tweets_data.columns]
    tweets_data.columns = new_columns

    tweetsdata_grouped = group_tweetsdata(tweets_data, time_interval)

    tweetsdata_grouped.to_csv('Data/2018tweets/Ready/Tweets(' + time_interval + ').csv')

    print("reading bitcoin_data")
    bitcoin_data = pd.read_csv("../raw_data/bitcoin/Bitstamp_BTCUSD_2018_minute.csv")
    bitcoin_data = bitcoin_data.sort_values(by='Date')
    bitcoin_data['Date'] = pd.to_datetime(bitcoin_data['Date'])
    bitcoin_data = bitcoin_data.loc[(bitcoin_data['Date'] >= start_date.replace(tzinfo=None))
                                    & (bitcoin_data['Date'] < end_date.replace(tzinfo=None))]

    print("reading volume_data")
    volume_data = pd.read_csv("../Data/2018tweets/2018(03-08--03-11).csv"
                              , lineterminator='\n')

    print("Saving DatasetBtcTweets")


if __name__ == '__main__':
    # preprocessing_mix()

    fill_weights()

    # language_filtering()
    # classify_tweets()
    # vec_tweets()

    create_dataset_mix(time_interval="60Min")
