import numpy as np
import pandas as pd
from time import time
import pandas as pd
import numpy as np
import re
import sys
import csv

import BotClassifier
# import ast
# def meanlist(list):
#     divider = len(list)
#     summed = list.pop()
#     while len(list) > 0:
#         summed = [sum(item) for item in zip(summed, list.pop())]
#
#     mean = []
#     for number in summed:
#         mean.append(number / divider)
#     return mean


if __name__ == '__main__':
    data = pd.read_csv("data/TweetsBTC_16mil/filtered/" +
                       "16_en_filtered.csv",
                       lineterminator='\n')

    data = data.sort_values(by='timestamp')

    data = data[["timestamp", "bert", "sent_neg", "sent_neu", "sent_pos", "bot"]]

    data = data[data.bot != True][["timestamp", "bert", "sent_neg", "sent_neu", "sent_pos"]]

    data['date'] = pd.to_datetime(data['timestamp'])
    data = data.set_index("timestamp")
    data.index = pd.to_datetime(data.index)

    head = data.head()
    head.head()

    aggregations = {
        'sent_neg': 'mean',
        'sent_neu': 'mean',
        'sent_pos': 'mean',
        "bert": lambda x: meanlist([ast.literal_eval(y) for y in x.values])
    }

    head.groupby(pd.Grouper(freq='10Min', offset="0m")).agg(aggregations).head()
