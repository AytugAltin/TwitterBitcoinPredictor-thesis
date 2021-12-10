import os
from DataProcessing.DataGrouper import group_tweetsdata, group_volumedata, group_bitcoindata, group_sentiment_volume, \
    group_sentiment_values
from FeatureExtraction import *


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

    print(" - Fixing BERT Features")
    final_df = fix_bert(potential_path)
    print(" - BERT Features Fixed", potential_path)

    print(" - Writing With BERT features")
    final_df.to_csv(potential_path)

    print(" ! Reading Tweets Data DONE...")
    return final_df


def load_compound_data(time_interval, files, renew=False):
    print("READING COMPOUND")
    potential_path = "../Data/2018-Weighted/cache/" + "COMPOUND" + "(" + time_interval + ")" + ".csv"

    if os.path.isfile(potential_path) and not renew:
        # Read CACHED FILE
        print(" - Cached File Found")
        df = pd.read_csv(potential_path)
        print(" - LOADED", potential_path)
        return df

    print(" - NO Cached File Found")

    root_path = "../Data/2018-Weighted/sentiment_values/"
    tweets_data = pd.DataFrame()
    for file_path in files:
        print("     - reading", file_path)
        temp = pd.read_csv(root_path + file_path)
        tweets_data = tweets_data.append(temp)

    print(" - Grouping")
    grouped_df = group_sentiment_values(tweets_data, time_interval)
    grouped_df = grouped_df[["sent_compound"]]
    print(" - Writing")
    grouped_df.to_csv(potential_path)

    return grouped_df


def fix_bert(path):
    bert = ""
    for i in range(0, 768):
        bert += ",bert" + str(i)

    bert += ","
    data = None
    with open(path, 'r') as infile:
        data = infile.read()
        data = data.replace('"', '')
        data = data.replace('[', '')
        data = data.replace(']', '')
        data = data.replace(',bert,', bert)
        infile.close()

    with open(path, 'w') as outfile:
        outfile.write(data)

    grouped_df = pd.read_csv(path)
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

    print(" ! Reading Volume Data DONE...")
    return grouped_df


def load_sentiment_volume_data(time_interval, files, renew=False):
    print("READING SENTIMENT VOLUME")
    potential_path = "../Data/2018-Weighted/cache/" + "SENT_VOLUME" + "(" + time_interval + ")" + ".csv"

    if os.path.isfile(potential_path) and not renew:
        # Read CACHED FILE
        print(" - Cached File Found")
        df = pd.read_csv(potential_path)
        print(" - LOADED", potential_path)
        return df

    print(" - NO Cached File Found")

    root_path = "../Data/2018-Weighted/sentiment_count/"
    sent_data = pd.DataFrame()
    for file_path in files:
        print("     - reading", file_path)
        temp = pd.read_csv(root_path + file_path)
        sent_data = sent_data.append(temp)

    print(" - Grouping")
    grouped_df = group_sentiment_volume(sent_data, time_interval)
    print(" - Writing")
    grouped_df.to_csv(potential_path)
    print(" - SENTIMENT VOLUME Data Cached in", potential_path)
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
    grouped_df["Date"] = grouped_df.index
    grouped_df = grouped_df.reset_index(drop=True)

    print(" ! Reading Bitcoin Data DONE...")
    return grouped_df
