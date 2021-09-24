import numpy as np

import pandas as pd
from whatthelang import WhatTheLang

def predictlang(text):
    try:
        return wtl.predict_lang(text)
    except:
        return "unknown"


if __name__ == '__main__':

    data = pd.read_csv("data/2021/Bitcoin_tweets.csv", sep=',')

    wtl = WhatTheLang()
    result = [predictlang(row) for row in data['text']]
    data['lang'] = result

    data.to_csv(path_or_buf="data/2021/Bitcoin_tweets_lang.csv")

    en_df = data[data["lang"] == 'en']
    en_df.dropna(how='any', inplace=True)

    en_df.to_csv(path_or_buf="data/2021/Bitcoin_tweets_en.csv")


