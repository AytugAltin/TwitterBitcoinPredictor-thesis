import pandas as pd
from whatthelang import WhatTheLang

wtl = WhatTheLang()
def predictlang(text):
    try:
        return wtl.predict_lang(text)
    except:
        return "unknown"


def filter_language(data):

    result = [predictlang(row) for row in data['text']]
    data['lang'] = result

    en_df = data[data["lang"] == 'en']
    en_df.dropna(how='any', inplace=True)

    return en_df, data
