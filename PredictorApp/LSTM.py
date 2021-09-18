import torch
from torch.utils.data import Dataset
from torchvision import datasets
import pandas as pd
from Datasets import *



dataset = DatasetBtcTweets("data/TweetsBTC_16mil/filtered/16_en_filtered.csv")

for data in dataset:
    print(data)
