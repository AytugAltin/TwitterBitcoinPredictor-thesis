import torch
from torch.utils.data import Dataset
from torchvision import datasets
import pandas as pd
import numpy as np


class DatasetBitcoin(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path, lineterminator='\n')

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data.iloc[index, 1:]
        label = self.data.iloc[index, 0]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DatasetCreator:
    def __init__(self):