import datetime

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from BotClassifier import BotClassifier
from Dataset import  DatasetBtcTweets
from Keys import BEARER_TOKEN
from TweetScraper import TweetScraper


def sentiment_scores(sentence):
    print("--", sentence, "--")
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)

    print("Overall sentiment dictionary is : ", sentiment_dict)
    print("sentence was rated as ", sentiment_dict['neg'] * 100, "% Negative")
    print("sentence was rated as ", sentiment_dict['neu'] * 100, "% Neutral")
    print("sentence was rated as ", sentiment_dict['pos'] * 100, "% Positive")

    print("Sentence Overall Rated As", end=" ")

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05:
        print("Positive")
    elif sentiment_dict['compound'] <= - 0.05:
        print("Negative")
    else:
        print("Neutral")


def sentiment_testing():
    sentences = [
        "This was a not good movie.",
    ]
    for sentence in sentences:
        sentiment_scores(sentence)
        print("")


def bot_testing():
    bot_clsfr = BotClassifier()

    print(bot_clsfr.tweet_is_bot("start trading bitcoin"))


def create_dataset(start_date, end_date):
    tweets_data = pd.read_csv("data/TweetsBTC_16mil/filtered/17_en_filtered.csv", lineterminator='\n')
    tweets_data["date"] = tweets_data["timestamp"]
    tweets_data = tweets_data.sort_values(by='date')
    data = tweets_data
    tweets_data = pd.read_csv("data/TweetsBTC_16mil/filtered/18_en_filtered.csv", lineterminator='\n')
    tweets_data["date"] = tweets_data["timestamp"]
    tweets_data = tweets_data.sort_values(by='date')
    data = data.append(tweets_data)
    tweets_data = pd.read_csv("data/TweetsBTC_16mil/filtered/19_en_filtered.csv", lineterminator='\n')
    tweets_data["date"] = tweets_data["timestamp"]
    tweets_data = tweets_data.sort_values(by='date')
    tweets_data = data.append(tweets_data)

    tweets_data['date'] = pd.to_datetime(tweets_data['date'])

    tweets_data = tweets_data.loc[
        (tweets_data['date'] >= start_date) &
        (tweets_data['date'] <= end_date)
        ]

    volume_data = pd.read_csv("data/TweetsBTC_16mil/peryear/17.csv", lineterminator='\n')
    volume_data["date"] = volume_data["timestamp"]
    volume_data = volume_data.sort_values(by='date')
    data = volume_data
    volume_data = pd.read_csv("data/TweetsBTC_16mil/peryear/18.csv", lineterminator='\n')
    volume_data["date"] = volume_data["timestamp"]
    volume_data = volume_data.sort_values(by='date')
    data = data.append(volume_data)
    volume_data = pd.read_csv("data/TweetsBTC_16mil/peryear/19.csv", lineterminator='\n')
    volume_data["date"] = volume_data["timestamp"]
    volume_data = volume_data.sort_values(by='date')
    volume_data = data.append(volume_data)

    volume_data['date'] = pd.to_datetime(volume_data['date'])

    volume_data = volume_data.loc[
        (volume_data['date'] >= start_date) &
        (volume_data['date'] <= end_date)
        ]

    bitcoin_data = pd.read_csv("data/bitcoin/Bitstamp_BTCUSD_2017_minute.csv")
    bitcoin_data = bitcoin_data.sort_values(by='Date')
    data = bitcoin_data
    bitcoin_data = pd.read_csv("data/bitcoin/Bitstamp_BTCUSD_2018_minute.csv")
    bitcoin_data = bitcoin_data.sort_values(by='Date')
    data = data.append(bitcoin_data)
    bitcoin_data = pd.read_csv("data/bitcoin/Bitstamp_BTCUSD_2019_minute.csv")
    bitcoin_data = bitcoin_data.sort_values(by='Date')
    bitcoin_data = data.append(bitcoin_data)

    bitcoin_data['Date'] = pd.to_datetime(bitcoin_data['Date'])

    bitcoin_data = bitcoin_data.loc[(bitcoin_data['Date'] >= start_date)
                                  & (bitcoin_data['Date'] < end_date)]

    dataset_creator = \
        DatasetBtcTweets(
            tweets_data=tweets_data,
            volume_data=volume_data,
            bitcoin_data=bitcoin_data,
            time_interval="60Min"
        )
    dataset_creator.print_status()


def scrape_tweets():
    start_time = datetime.datetime(2021, 2, 5, 0, 0, 0, 0, datetime.timezone.utc)
    end_time = datetime.datetime(2021, 2, 6, 0, 0, 0, 0, datetime.timezone.utc)

    scraper = TweetScraper(BEARER_TOKEN)

    data = scraper.get_tweets(start_time=start_time, end_time=end_time, tags="bitcoin OR btc")

    data.to_csv(path_or_buf="data/scraped/tweets_" + str(start_time.date()) + str(end_time.date()) + ".csv")


if __name__ == '__main__':
    # sentiment_testing()
    # bot_testing()
    create_dataset(start_date=datetime.datetime(2017, 1, 1, 0, 0, 0, 0, datetime.timezone.utc),
                   end_date=datetime.datetime(2019, 3, 28, 0, 0, 0, 0, datetime.timezone.utc)
                   )
    # scrape_tweets()
