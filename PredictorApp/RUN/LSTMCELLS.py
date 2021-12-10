import pandas as pd
import torch
from torch import optim

from Models.EXPMODEL import EXPMODEL
from Models.LSTMCells import LSTMCELLS
from Training.LossFunctions import LOSS_FUNCTIONS
from Training.Main import train_main
from Device import DEVICE

TIME_INTERVALS = [
    # "10Min",# "30Min",
    # "1H",# "2H",
    #  "3H",
    # "6H",
    # "12H",
    "24H"
]
BATCH_SIZE = 32
BERT_ENABLE = False
EPOCHS = 1000

TIME_DIFFERENCE_ENABLE = False
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
MAX_NO_IMPROVEMENTS = 10

LOSS = LOSS_FUNCTIONS["MSE"]

WINDOW_SIZES = [
    # 1,
    # 2,
    5,
    # 10,
    # 25,
    # 50,
    # 100
]
SETS = [
    # 1,
    # 2,
    # 3,
    4
]
MODEL_NAME = "EXPMODEL"

BERT_SIZE = 768
FEATURES = [
    # 'Volume',
    # 'tweet_vol',
    'sent_neg',
    'sent_neu',
    'sent_pos',
    'Close'
]
TO_PREDICT_LABELS = ["Close"]

TEST_RATIO = 0.05
VAL_RATIO = 0.1

HIDDEN_DIM = 8
NUM_LAYERS = 1
DROPOUT = 0.2

if __name__ == '__main__':
    df_final_results = pd.DataFrame()
    for window_size in WINDOW_SIZES:
        input_dim = int(len(FEATURES) * window_size)
        model = LSTMCELLS(input_dim=input_dim, hidden_dim=HIDDEN_DIM,
                          output_dim=1,dropout=DROPOUT,
                          batch_size=BATCH_SIZE).to(DEVICE)

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
