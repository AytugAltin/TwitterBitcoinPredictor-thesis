import numpy as np

import pandas as pd
from whatthelang import WhatTheLang
import contractions as contractions
import unicodedata

if __name__ == '__main__':
    string = "Hey İbrahim Nas thanks for the follow!\r"
    string = string.lower()

    unicodedata.normalize('NFD', string).encode('ascii', 'ignore').decode("utf-8")
    x = ' '.join([contractions.fix(word) for word in string.split()])


