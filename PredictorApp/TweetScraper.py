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

        search_results = self.twarc.search_all(query=tags, start_time=start_time, end_time=end_time,
                                               max_results=100)
        data = pd.DataFrame({'A': []})
        log = 10
        write = 100
        index = 0
        for page in search_results:
            # for tweet in ensure_flattened(page):
            if data.empty:
                data = self.page_to_dataframe(page)
            else:
                new_data = self.page_to_dataframe(page)
                data = data.append(new_data)
            if index % log == 0:
                print(index, " Data_Shape", data.shape)
                print(data.iloc[-1]["created_at"])

            if index % write == 0:
                print("__WRITING",)
                data.to_csv(path_or_buf="data/scraped/tweets_" + str(start_time.date()) + str(end_time.date()) + ".csv")

            index = index + 1

        return data

    def page_to_dataframe(self, page):
        return pd.DataFrame(ensure_flattened(page))



