import datetime

import pandas as pd
import tweepy
from Keys import *
from twarc.client2 import Twarc2
from twarc.expansions import ensure_flattened


class TweetScraper:
    def __init__(self, BEARER_TOKEN):
        self.twarc = Twarc2(bearer_token=BEARER_TOKEN)

    def get_tweets(self, start_time, end_time, tags):
        path = "raw_data/scraped/tweets_" + str(start_time.date()) + str(end_time.date()) + ".csv"
        print("####Scraping to", path)

        search_results = self.twarc.search_all(query=tags, start_time=start_time, end_time=end_time,
                                               max_results=100)

        # data_raw = pd.DataFrame()
        data_processed = pd.DataFrame()
        log = 10
        write = 100
        index = 0
        for page in search_results:
            # data_raw = data_raw.append(self.page_to_dataframe(page))
            for tweet in ensure_flattened(page):
                data_processed = data_processed.append(self.proces_tweet(tweet), ignore_index=True)
            if index % log == 0:
                print(index, " Data_Shape", data_processed.shape)
                print(data_processed.iloc[-1]["date"])

            if index % write == 0:
                print("__WRITING", )
                data_processed.to_csv(
                    path_or_buf=path)

            index = index + 1
        print("Saved To ", path)
        return data_processed

    def page_to_dataframe(self, page):
        return pd.DataFrame(ensure_flattened(page))

    def proces_tweet(self, tweet):
        return {
            "text": tweet["text"],
            "date": tweet["created_at"],
            "api_lang": tweet["lang"],
            "possibly_sensitive": tweet['possibly_sensitive'],
            'source': tweet['source'],
            'retweet_count': tweet['public_metrics']["retweet_count"],
            'reply_count': tweet['public_metrics']["reply_count"],
            'like_count': tweet['public_metrics']["like_count"],
            'quote_count': tweet['public_metrics']["quote_count"],
            'author_id': tweet['author_id'],
            'conversation_id': tweet['conversation_id'],
            'id': tweet['id'],
            'username': tweet['author']['username'],
            'user_created_at': tweet['author']['created_at'],
            'user_id': tweet['author']['id'],
            'user_verified': tweet['author']['verified'],
            'user_followers_count': tweet['author']['public_metrics']['followers_count'],
            'user_following_count': tweet['author']['public_metrics']['following_count'],
            'user_tweet_count': tweet['author']['public_metrics']['tweet_count'],
            'user_listed_count': tweet['author']['public_metrics']['listed_count'],
        }
