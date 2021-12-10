import pandas as pd
import torch
from torch import optim

from Models.EXPMODEL import EXPMODEL
from Models.LSTM.LSTMModel import LSTMModel
from Training.LossFunctions import LOSS_FUNCTIONS
from Training.Main import train_main
from Device import DEVICE

TIME_INTERVALS = [
    "10Min", "30Min", "1H", "2H",
    "3H", "6H", "12H",
    "24H"]
BATCH_SIZE = 64
BERT_ENABLE = False
EPOCHS = 1000

TIME_DIFFERENCE_ENABLE = True

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
MAX_NO_IMPROVEMENTS = 15
WINDOW_SIZES = [
    2,
    5,
    10,
    25,
    50,
    100
]
SETS = [
    1,
    2,
    3,
    4
]

BERT_SIZE = 768

FEATURES = [
    # 'Volume',
    # 'tweet_vol',
    'sent_neg',
    'sent_neu',
    'sent_pos',
    # "sent_compound",
    # 'count_neg',
    # 'count_neu',
    # 'count_pos',
    'Close',
    # "High",
    # "Low",
]
TO_PREDICT_LABELS = ["Close"]

LOSS = LOSS_FUNCTIONS["MSE"]

TEST_RATIO = 0.05
VAL_RATIO = 0.1

HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.2

if __name__ == '__main__':
    df_final_results = pd.DataFrame()
    for window_size in WINDOW_SIZES:
        input_dim = int(len(FEATURES) * window_size)
        model = LSTMModel(input_dim, HIDDEN_DIM,
                          NUM_LAYERS, 1,
                          DROPOUT, BATCH_SIZE).to(DEVICE)

        optimizer = optim.Adam(model.parameters(),
                               lr=LEARNING_RATE,
                               weight_decay=WEIGHT_DECAY)

        df_final_results = \
            train_main(df_final_results, TIME_INTERVALS,
                       window_size, model, optimizer, LOSS,
                       TIME_DIFFERENCE_ENABLE, FEATURES,
                       BERT_ENABLE, BERT_SIZE,
                       SETS, BATCH_SIZE,
                       MAX_NO_IMPROVEMENTS, EPOCHS)

    df_final_results.to_csv(path_or_buf="../Logs/SUMMARY" + ".csv")
