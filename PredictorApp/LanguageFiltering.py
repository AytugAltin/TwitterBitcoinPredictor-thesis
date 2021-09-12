import numpy as np

import pandas as pd
from whatthelang import WhatTheLang

if __name__ == '__main__':

    sets = ["16",
            "17",
            "18",
            "19"
            ]

    for setName in sets:
        data = pd.read_csv("data/TweetsBTC_16mil/peryear/" + setName + ".csv",
                           lineterminator='\n')


        wtl = WhatTheLang()
        result = [wtl.predict_lang(row) for row in data['text']]
        data['lang'] = result
        data.head()

        en_df = data[data["lang"] == 'en']
        en_df.dropna(how='any', inplace=True)

        en_df.to_csv(path_or_buf="data/TweetsBTC_16mil/" + setName + "_en.csv")


