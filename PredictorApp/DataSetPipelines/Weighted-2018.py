import datetime
import pandas as pd
import time
from DataProcessing.CalculateBERT import vectorize_tweets
from DataProcessing.ClassifyTweets import classify
import os
import pickle
from DataProcessing.DataGrouper import *


def classify_tweets(files):
    print("classify_tweets")
    root_path = "../Data/2018-Weighted/unprocessed/"

    for file in files:
        try:
            data = pd.read_csv(root_path + file + ".csv", sep=";")
        except:
            data = pd.read_csv(root_path + file + ".csv", lineterminator="\n")
        print("Loaded", file, " ", data.shape)

        data = data.loc[data['api_lang'] == "en"]

        data = classify(data)
        data.to_csv(path_or_buf="../Data/2018-Weighted/classed/" + file + ".csv")


def vec_tweets(files):
    print("vec_tweets")
    root_path = "../Data/2018-Weighted/classed/"

    chunksize = 100000

    for file in files:
        try:
            df_chunk = pd.read_csv(root_path + file + ".csv", lineterminator="\n", chunksize=chunksize)
        except:
            df_chunk = pd.read_csv(root_path + file + ".csv", sep=";", chunksize=chunksize)

        t0 = time.time()
        index = 0
        chunk_list = []
        for chunk in df_chunk:
            print("vectorize", chunk.shape, " time: ", (time.time() - t0))
            chunk_vecced = vectorize_tweets(chunk)
            index = index + 1
            chunk_list.append(chunk_vecced)
            print("shape", chunk.shape, " time: ", (time.time() - t0))

        print("Written", file)
        df_concat = pd.concat(chunk_list)
        df_concat.to_csv("../Data/2018-Weighted/bert/" + file + ".csv")


def pre_group_data(files, time_interval):
    print("group_tweets")
    root_path = "../Data/2018-Weighted/bert/"

    chunksize = 100000

    for file in files:
        try:
            df_chunk = pd.read_csv(root_path + file + ".csv", lineterminator="\n", chunksize=chunksize)
        except:
            df_chunk = pd.read_csv(root_path + file + ".csv", sep=";", chunksize=chunksize)

        t0 = time.time()
        index = 0
        chunk_list = []
        for chunk in df_chunk:
            print("group_tweetsdata", chunk.shape, " time: ", (time.time() - t0))

            new_columns = [i.strip() for i in chunk.columns]
            chunk.columns = new_columns

            chunk["weight"] = chunk["user_followers_count"] + 1
            chunk["count"] = 1

            chunk_grouped = group_tweetsdata(chunk, time_interval)

            chunk_list.append(chunk_grouped)

            print("shape", chunk.shape, " index", index, " time: ", (time.time() - t0))
            index = index + 1

        print("Written", index, file)
        df_concat = pd.concat(chunk_list)
        df_concat.to_csv("../Data/2018-Weighted/grouped/" + file + "(" + time_interval + ")" ".csv")


if __name__ == '__main__':
    # files = [
    #     "03 2018", "04 2018", "05 2018",
    #     "06 2018",
    #     "07 2018", "08 2018",
    #     "09 2018", "10 2018",
    #     "11 2018"
    # ]
    #
    # classify_tweets(files)

    # files = [
    #     # "03 2018",
    #     "04 2018",
    #     # "05 2018",
    #     # "06 2018",
    #     # "07 2018",
    #     # "08 2018","09 2018", "10 2018", "11 2018"
    # ]
    #
    # vec_tweets(files)

    files = [
        "03 2018", "04 2018",
        "05 2018", "06 2018", "07 2018",
        "08 2018", "09 2018", "10 2018",
        "11 2018"
    ]

    pre_group_data(files, "1Min")

    # todo for the 6th month
