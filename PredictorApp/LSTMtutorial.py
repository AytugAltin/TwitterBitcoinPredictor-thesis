import torch
import torch.nn as nn
from Dataset import *

INPUT_FEATURES = [
    'Volume',
    'tweet_vol',
    'sent_neg', 'sent_neu', 'sent_pos',
    # 'bert',
    # 'count'
]

OUTPUT_FEATURES = ['Close']
if __name__ == '__main__':
    dataset = CombinedDataset(csv_file="Data/2018tweets/Objects/(60Min).csv",
                              input_features=INPUT_FEATURES, output_features=OUTPUT_FEATURES)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(dataset.data, INPUT_FEATURES,
                                                                          OUTPUT_FEATURES, 0.2)

    df = dataset.data

    input_size = len(X_train.columns)
    output_dim = 1
    hidden_size = 128
    num_layers = 4
    dropout = 0.6
    n_epochs = 100
    learning_rate = 1e-1
    weight_decay = 1e-6

    lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)  # Input dim is 3, output dim is 3
