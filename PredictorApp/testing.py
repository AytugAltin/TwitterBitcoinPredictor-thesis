import numpy as np
import pandas as pd
from time import time
import pandas as pd
import numpy as np
import re
import sys
import csv

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
    data = pd.read_csv("data/2021/Bitcoin_tweets.csv")
    data = data.sort_values(by='date')
    data = data[["date"]]

    data = data.set_index("date")
    data.index = pd.to_datetime(data.index)
    grouped = data.groupby(pd.Grouper(freq="60Min")).size().reset_index(name='tweet_vol')
    data.head(30)
