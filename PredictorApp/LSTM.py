import torch
from torch.utils.data import Dataset
from torchvision import datasets
import pandas as pd
from Dataset import *
import pickle
import datetime

with open('Data/2018tweets/Objects/(60Min).pickle', 'rb') as handle:
    dataset = pickle.load(handle)

for data in dataset:
    print(data)

num_layers = 1
learning_rate = 0.005
size_layer = 128
timestamp = 5
epoch = 500
dropout_rate = 0.6