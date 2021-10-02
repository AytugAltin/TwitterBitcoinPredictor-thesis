import numpy as np

import pandas as pd
from whatthelang import WhatTheLang


def predictlang(text):
    try:
        return wtl.predict_lang(text)
    except:
        return "unknown"


if __name__ == '__main__':

    file_name = "data/mix/Per Period/2018(03-08--03-11)"
    file_path = file_name + ".csv"
    fields = ["date", "text"]

    try:
        data = pd.read_csv(file_path, sep=";",usecols = fields)
    except:
        data = pd.read_csv(file_path, lineterminator="\n",usecols = fields)

    wtl = WhatTheLang()
    result = [predictlang(row) for row in data['text']]
    data['lang'] = result

    data.to_csv(path_or_buf=file_name + "_lang.csv")

    en_df = data[data["lang"] == 'en']
    en_df.dropna(how='any', inplace=True)

    en_df.to_csv(path_or_buf=file_name + "_en.csv")
