import torch
from torch.utils.data import Dataset
from torchvision import datasets
import pandas as pd
from Dataset import *
import pickle

som = DatasetBtcTweets()
fileObject = "path"
pickle.dump(som, fileObject)
#...
som = pickle.load(fileObject)
som.work()




for data in dataset:
    print(data)
