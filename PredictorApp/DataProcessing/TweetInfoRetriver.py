import pandas as pd
from twarc.client2 import Twarc2
from twarc.expansions import ensure_flattened
import datetime
from TwitterAPI.Keys import BEARER_TOKEN
from datetime import timedelta
from TwitterAPI.TweetScraper import proces_tweet


def add_info_to_tweets(df):
    print(BEARER_TOKEN)
    twarc = Twarc2(bearer_token=BEARER_TOKEN)
    count_found = 0
    count_not_found = 0

    ids = df["id"]
    lookup_results = twarc.tweet_lookup(ids)
    data_processed = pd.DataFrame()
    index = 0
    log = 10
    for page in lookup_results:
        for tweet in ensure_flattened(page):
            data_processed = data_processed.append(proces_tweet(tweet), ignore_index=True)


        if index % log == 0:
            print(index, " Data_Shape", data_processed.shape)
            print(data_processed.iloc[-1]["date"])

        index = index + 1

    return data_processed



def set_values(df, index, tweet_info):
    list = [
        "id",
        "api_lang",
        "possibly_sensitive",
        'source',
        'retweet_count',
        'reply_count',
        'like_count',
        'quote_count',
        'author_id',
        'conversation_id',
        'user_created_at',
        'user_id',
        'user_verified',
        'user_followers_count',
        'user_following_count',
        'user_tweet_count',
        'user_listed_count',
        'user_description'
    ]

    for tag in list:
        df._set_value(index, tag, tweet_info[tag])
