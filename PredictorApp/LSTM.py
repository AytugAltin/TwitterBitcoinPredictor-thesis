import torch
from torch.utils.data import Dataset
from torchvision import datasets
import pandas as pd
from Datasets import *

csv = pd.read_csv("data/SentimentBTC_17mil/df_Final.csv",
                  names=["Date", "Total Volume of Tweets", "Count_Negatives", "Count_Positives",
                         "Count_Neutrals", "Sent_Negatives", "Sent_Positives", "Count_News", "Count_Bots", "Open",
                         "High", "Low", "Close", "Volume (BTC)", "Volume (Currency)"
                         ],
                  keep_default_na=False
                  )  # filtered

0
dataset = DatasetBitcoin("data/TweetsBTC_16mil/tweets.csv")

for data in dataset:
    print(data)
