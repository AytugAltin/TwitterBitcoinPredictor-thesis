import datetime
from time import strftime

import pandas as pd
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from DataLoader import *
from Device import DEVICE
from Evaluate import evaluate_model
from Models.Hybrid.CNNLSTM import CNNLSTM
from Models.Hybrid.DenseLSTM import DenseLSTM
from Models.LSTM.LSTM import LSTM
from Trainer import Trainer
import csv
import numpy as np

from torchvision import models
from torchsummary import summary

# region CONFIG
TIME_INTERVALS = [
    # "10Min", "30Min", "1H", "2H",
    # "3H", "6H", "12H",
    "24H"]
BATCH_SIZE = 64
BERT_ENABLE = False
EPOCHS = 1000

TIME_DIFFERENCE_ENABLE = True

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
WINDOW_SIZES = [
    10, 25, 50,
    100
]
MODEL_NAME = "LSTM"
# MODEL_NAME = "CNNLSTM"
# MODEL_NAME = "DenseLSTM"

BERT_SIZE = 768
FEATURES = [
    # 'Volume',
    # 'tweet_vol',
    # 'sent_neg',
    # 'sent_neu',
    # 'sent_pos',
    'Close'
]
TO_PREDICT_LABELS = ["Close"]

TEST_RATIO = 0.05
VAL_RATIO = 0.1


# endregion

# region HELPERS
def time_differencing(df):
    df["Close_v"] = df["Close"]
    df = time_difference(df, "Close")
    return df


def time_differencing_reverse(df, zero_value):
    x, x_diff = zero_value, df['value'].iloc[:]
    df['value_real'] = np.r_[x, x_diff].cumsum().astype(float)[1:]
    x, x_diff = zero_value, df['prediction'].iloc[:]
    df['prediction_real'] = np.r_[x, x_diff].cumsum().astype(float)[1:]
    return df


def feature_extration(df_raw, window_size):
    print("EXTRACTING FEATURES")
    df_result = time_lag_features(df_raw, FEATURES, N=window_size)
    print("TIME LAGGING FEATURES DONE")

    bert_vector = []
    if BERT_ENABLE:
        print("ADDING BERT FEATURES")
        for i in range(0, BERT_SIZE):
            bert_vector.append("bert" + str(i))

        df_bert = df_raw[bert_vector]
        df_bert = df_bert.iloc[window_size:, :]  # Compensate the time lag
        df_result = pd.concat([df_result, df_bert], axis=1)

        # df_bert = create_bert_feature(df_bert,WINDOW_SIZE)

    return df_result


def cross_validation_sets(df):
    start_date = datetime.datetime(2018, 3, 8, 0, 0, 0, 0, datetime.timezone.utc)

    end_dates = [datetime.datetime(2018, 7, 1, 0, 0, 0, 0, datetime.timezone.utc),
                 datetime.datetime(2018, 8, 1, 0, 0, 0, 0, datetime.timezone.utc),
                 datetime.datetime(2018, 9, 1, 0, 0, 0, 0, datetime.timezone.utc),
                 datetime.datetime(2018, 10, 1, 0, 0, 0, 0, datetime.timezone.utc),
                 datetime.datetime(2018, 11, 4, 0, 0, 0, 0, datetime.timezone.utc),
                 ]

    df.index = pd.to_datetime(df.index)
    end_date = end_dates.pop(0)

    sets = []
    set_index = 1
    while len(end_dates) > 0:
        end_date_train = end_date - datetime.timedelta(days=15)

        training_set = df.loc[(df.index >= start_date.replace(tzinfo=None))
                              & (df.index < end_date_train.replace(tzinfo=None))]
        print("SET", set_index)
        print("Train", "start", str(start_date), "end", str(end_date_train))

        validation_set = df.loc[(df.index >= end_date_train.replace(tzinfo=None))
                                & (df.index < end_date.replace(tzinfo=None))]

        print("validate", "start", str(end_date_train), "end", str(end_date))

        start_date_test = end_date
        end_date = end_dates.pop(0)

        test_set = df.loc[(df.index >= start_date_test.replace(tzinfo=None))
                          & (df.index < end_date.replace(tzinfo=None))]

        print("test", "start", str(start_date_test), "end", str(end_date))

        sets.append((training_set, validation_set, test_set, set_index))
        print("------")
        set_index += 1

    return sets


def create_data_loaders(X_train, X_val, y_train, y_val, X_test, y_test, scaler):
    # from https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b

    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)

    y_train_arr = scaler.fit_transform(y_train)
    y_val_arr = scaler.transform(y_val)
    y_test_arr = scaler.transform(y_test)

    train_features = torch.Tensor(X_train_arr).to(DEVICE)
    train_targets = torch.Tensor(y_train_arr).to(DEVICE)
    val_features = torch.Tensor(X_val_arr).to(DEVICE)
    val_targets = torch.Tensor(y_val_arr).to(DEVICE)
    test_features = torch.Tensor(X_test_arr).to(DEVICE)
    test_targets = torch.Tensor(y_test_arr).to(DEVICE)

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    if BATCH_SIZE > len(train):
        train_loader = DataLoader(train, batch_size=len(train), shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    if BATCH_SIZE > len(val):
        val_loader = DataLoader(val, batch_size=len(val), shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, test_loader_one


def dict_to_file(dict, path):
    w = csv.writer(open(path, "w"))
    for key, val in dict.items():
        w.writerow([key, val])


def get_result_df(dict):
    df_r = pd.DataFrame.from_dict(dict.items()).T
    df_r.columns = df_r.iloc[0]
    df_r = df_r.drop(df_r.index[0])
    return df_r


def final_results(trainer, train_loader, val_loader, test_loader,
                  X_test, X_val, X_train, scaler, root,
                  time_interval, window_size, set_index,
                  zero_value
                  ):
    df_result_train, result_metrics_train = evaluate_model(model=trainer.best_model, test_loader=train_loader,
                                                           batch_size=train_loader.batch_size,
                                                           n_features=trainer.num_features, X_test=X_train,
                                                           scaler=scaler)

    df_result_train = time_differencing_reverse(df_result_train, zero_value)
    zero_value = df_result_train["value_real"][-1]

    df_result_val, result_metrics_val = evaluate_model(model=trainer.best_model, test_loader=val_loader,
                                                       batch_size=val_loader.batch_size,
                                                       n_features=trainer.num_features, X_test=X_val,
                                                       scaler=scaler)

    df_result_val = time_differencing_reverse(df_result_val, zero_value)
    zero_value = df_result_val["value_real"][-1]

    df_result_test, result_metrics_test = evaluate_model(model=trainer.best_model, test_loader=test_loader,
                                                         batch_size=test_loader.batch_size,
                                                         n_features=trainer.num_features, X_test=X_test,
                                                         scaler=scaler)
    df_result_test = time_differencing_reverse(df_result_test, zero_value)

    details = "TI" + str(time_interval) + "WS" + str(window_size) + "SET" + str(set_index)

    df_result_train.to_csv(path_or_buf=root + "-results_train" + details + ".csv")
    df_result_val.to_csv(path_or_buf=root + "-results_val" + details + ".csv")
    df_result_test.to_csv(path_or_buf=root + "-results_test" + details + ".csv")

    title = "TI=" + str(time_interval) + "    /WS=" + str(window_size) + "  /SET=" + str(set_index)

    trainer.plot_predictions(test_loader_one=test_loader, batch_size=1, n_features=trainer.num_features,
                             X_test=X_test, scaler=scaler, model=trainer.best_model, title=title)

    dict = {
        "LearningRate": LEARNING_RATE,
        "WeightDecay": WEIGHT_DECAY,
        "BatchSize": BATCH_SIZE,

        "TimeInterval": time_interval,
        "WindowSize": window_size,
        "SetIndex": set_index,

        "Features": str(FEATURES),
        "BertEnable": str(BERT_ENABLE),

        "TimeDifferenceEnable": str(TIME_DIFFERENCE_ENABLE),

        "TrainingSetSize": len(train_loader.dataset),
        "ValidationSetSize": len(val_loader.dataset),
        "TestSetSize": len(test_loader.dataset),

        "TrainingMAE": round(result_metrics_train["mae"], 2),
        "TrainingRMSE": round(result_metrics_train["rmse"], 2),
        "TrainingMSE": round(result_metrics_train["mse"], 2),
        "TrainingR": round(result_metrics_train["r2"], 4),

        "ValidationMAE": round(result_metrics_val["mae"], 2),
        "ValidationRMSE": round(result_metrics_val["rmse"], 2),
        "ValidationMSE": round(result_metrics_val["mse"], 2),
        "ValidationR": round(result_metrics_val["r2"], 4),

        "TestMAE": round(result_metrics_test["mae"], 2),
        "TestRMSE": round(result_metrics_test["rmse"], 2),
        "TestMSE": round(result_metrics_test["mse"], 2),
        "TestR": round(result_metrics_test["r2"], 4),

        "EpochEnd": trainer.epoch,

        "ModelName": MODEL_NAME,
    }
    model_dict = trainer.best_model.get_model_dict_info()
    dict.update(model_dict)
    dict_to_file(dict=dict, path=root + "-SUMMARY" + ".csv")
    return get_result_df(dict)


# endregion

def train_main(time_interval, window_size, df_final_results):
    current = strftime('_%d_%H%M')
    # region Load_Data
    files = [
        "03 2018(1Min).csv",
        "04 2018(1Min).csv",
        "05 2018(1Min).csv",
        "06 2018(1Min).csv",
        "07 2018(1Min).csv",
        "08 2018(1Min).csv",
        "09 2018(1Min).csv",
        "10 2018(1Min).csv",
        "11 2018(1Min).csv"
    ]
    start_date = datetime.datetime(2018, 3, 8, 0, 0, 0, 0, datetime.timezone.utc)
    end_date = datetime.datetime(2018, 11, 4, 0, 0, 0, 0, datetime.timezone.utc)

    dataframes = []

    tweet_df = load_tweet_data(time_interval, files)
    dataframes.append(tweet_df)

    volume_df = load_volume_data(time_interval, start_date, end_date)
    dataframes.append(volume_df)

    bitcoin_df = load_bitcoin_data(time_interval, start_date, end_date)
    dataframes.append(bitcoin_df)

    df_raw = pd.DataFrame()
    for df in dataframes:
        df_raw = pd.concat([df_raw, df], axis=1)

    df_raw = df_raw.set_index('Date')
    print("DATA READY")

    zero_value = 0
    if TIME_DIFFERENCE_ENABLE:
        print("TIME DIFFERENCE ENABLED")
        zero_value = df_raw["Close"][window_size - 1]
        df_raw = time_differencing(df_raw)

    df_result = feature_extration(df_raw, window_size)
    print(" CROSS VALIDATION AND SPLITTING DATASET")

    sets = cross_validation_sets(df_result)
    sets.reverse()
    num_features = len(df_result.columns) - len(TO_PREDICT_LABELS)
    try:
        for (training_set, validation_set, test_set, set_index) in sets:
            # region SetupLoaders

            X_train, y_train = feature_label_split(training_set, TO_PREDICT_LABELS)
            X_val, y_val = feature_label_split(validation_set, TO_PREDICT_LABELS)
            X_test, y_test = feature_label_split(test_set, TO_PREDICT_LABELS)
            scaler = MinMaxScaler()
            train_loader, val_loader, test_loader, test_loader_one = \
                create_data_loaders(X_train, X_val, y_train, y_val, X_test, y_test, scaler)

            # endregion

            # region MODEL
            input_dim = num_features
            hidden_dim = 128
            num_layers = 3
            output_dim = 1
            dropout = 0.2

            Models = {}
            try:
                Models["LSTM"] = LSTM(input_dim, hidden_dim, num_layers, output_dim,
                                      dropout, BATCH_SIZE).to(DEVICE)
                Models["CNNLSTM"] = CNNLSTM(input_dim, hidden_dim, num_layers, output_dim,
                                            dropout, BATCH_SIZE, BERT_SIZE, window_size).to(DEVICE)
                Models["DenseLSTM"] = DenseLSTM(input_dim, hidden_dim, num_layers, output_dim,
                                                dropout, BATCH_SIZE, BERT_SIZE, window_size).to(DEVICE)
            except:
                pass

            model = Models[MODEL_NAME]

            print(model)

            # endregion

            # region Root_NAME
            root = "../Logs/"
            if not os.path.exists(root):
                os.makedirs(root)

            root = root + "TI(" + str(time_interval) + ")/"
            if not os.path.exists(root):
                os.makedirs(root)

            root = root + "WS(" + str(window_size) + ")/"
            if not os.path.exists(root):
                os.makedirs(root)

            root = root + "SET" + str(set_index) + "/"
            if not os.path.exists(root):
                os.makedirs(root)

            # endregion

            name = ""

            # region Summary
            print("SUMMARY")
            if BERT_ENABLE:
                features = [x for x in FEATURES if not x.startswith('bert')]
                features.append("BERT_VECTOR")
            else:
                features = FEATURES

            print("- Features", features)
            print("- TIME_INTERVAL", time_interval, "/ SET", set_index, "/ WINDOW SIZE", window_size, "/ BATCH_SIZE",
                  BATCH_SIZE)
            print("- Epochs", EPOCHS, "/ LEARNING_RATE", LEARNING_RATE, "/ WEIGHT_DECAY", WEIGHT_DECAY)
            print("- LSTM Details: ", "hidden_dim", hidden_dim, "/ num_layers", num_layers, "/ dropout", dropout)
            # endregion

            print("STARTING TRAINING", features)
            loss = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            trainer = Trainer(model=model, loss_fn=loss, optimizer=optimizer, num_features=input_dim, name=name,
                              root=root)

            trainer.train(train_loader, val_loader, n_epochs=EPOCHS, test_loader_one=test_loader_one,
                          X_test=X_test, X_val=X_val, X_train=X_train, scaler=scaler)

            print("MODEL DONE TRAINING")
            df_r = final_results(trainer=trainer, train_loader=train_loader, val_loader=val_loader,
                                 test_loader=test_loader_one,
                                 X_test=X_test, X_val=X_val, X_train=X_train, scaler=scaler,
                                 root=root, time_interval=time_interval, window_size=window_size, set_index=set_index,
                                 zero_value=zero_value)

            df_final_results = df_final_results.append(df_r)
    except Exception as e:
        print(e)

    return df_final_results


if __name__ == '__main__':
    for time_interval in TIME_INTERVALS:
        df_final_results = pd.DataFrame()
        for window_size in WINDOW_SIZES:
            df_final_results = train_main(time_interval, window_size, df_final_results)

        df_final_results.to_csv(path_or_buf="../Logs/SUMMARY_" + str(time_interval) + ".csv")
