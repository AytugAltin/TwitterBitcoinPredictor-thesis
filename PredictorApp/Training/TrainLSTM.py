import datetime
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from DataLoader import *
from Device import DEVICE
from Models.Hybrid.CNNLSTM import CNNLSTM
from Models.Hybrid.DenseLSTM import DenseLSTM
from Models.LSTM.LSTM import LSTM
from Trainer import Trainer

# region CONFIG
TIME_INTERVAL = "6H"
BATCH_SIZE = 64
BERT_ENABLE = False
EPOCHS = 10000

TIME_DIFFERENCE_ENABLE = False

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
WINDOW_SIZE = 25
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

def time_differencing(df):
    df = time_difference(df_raw, "Close")
    return df


def feature_extration(df_raw):
    print("EXTRACTING FEATURES")
    df_result = time_lag_features(df_raw, FEATURES, N=WINDOW_SIZE)
    print("TIME LAGGING FEATURES DONE")

    bert_vector = []
    if BERT_ENABLE:
        print("ADDING BERT FEATURES")
        for i in range(0, BERT_SIZE):
            bert_vector.append("bert" + str(i))

        df_bert = df_raw[bert_vector]
        df_bert = df_bert.iloc[WINDOW_SIZE:, :]  # Compensate the time lag
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
    while len(end_dates) > 0:
        training_set = df.loc[(df.index >= start_date.replace(tzinfo=None))
                              & (df.index < end_date.replace(tzinfo=None))]

        start_date_test = end_date
        end_date = end_dates.pop(0)

        test_set = df.loc[(df.index >= start_date_test.replace(tzinfo=None))
                          & (df.index < end_date.replace(tzinfo=None))]

        sets.append((training_set, test_set))

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
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    if BATCH_SIZE > len(val):
        val_loader = DataLoader(val, batch_size=len(val), shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, test_loader_one


if __name__ == '__main__':
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

    tweet_df = load_tweet_data(TIME_INTERVAL, files)
    dataframes.append(tweet_df)

    volume_df = load_volume_data(TIME_INTERVAL, start_date, end_date)
    dataframes.append(volume_df)

    bitcoin_df = load_bitcoin_data(TIME_INTERVAL, start_date, end_date)
    dataframes.append(bitcoin_df)

    df_raw = pd.DataFrame()
    for df in dataframes:
        df_raw = pd.concat([df_raw, df], axis=1)

    df_raw = df_raw.set_index('Date')
    print("DATA READY")

    if TIME_DIFFERENCE_ENABLE:
        print("TIME DIFFERENCE ENABLED")
        df_raw = time_differencing(df_raw)

    df_result = feature_extration(df_raw)
    print("SPLITTING DATASET")

    sets = cross_validation_sets(df_result)
    sets.reverse()
    num_features = len(df_result.columns) - len(TO_PREDICT_LABELS)

    for (training_set, test_set) in sets:
        # training set
        X, y = feature_label_split(training_set, TO_PREDICT_LABELS)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_RATIO, shuffle=False)
        # test set
        X_test, y_test = feature_label_split(test_set, TO_PREDICT_LABELS)
        scaler = MinMaxScaler()
        train_loader, val_loader, test_loader, test_loader_one = \
            create_data_loaders(X_train, X_val, y_train, y_val, X_test, y_test, scaler)

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
                                        dropout, BATCH_SIZE, BERT_SIZE, WINDOW_SIZE).to(DEVICE)
            Models["DenseLSTM"] = DenseLSTM(input_dim, hidden_dim, num_layers, output_dim,
                                            dropout, BATCH_SIZE, BERT_SIZE, WINDOW_SIZE).to(DEVICE)
        except:
            pass

        model = Models["LSTM"]

        # endregion

        loss = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        opt = Trainer(model=model, loss_fn=loss, optimizer=optimizer, num_features=input_dim)

        print("SUMMARY")

        if BERT_ENABLE:
            features = [x for x in FEATURES if not x.startswith('bert')]
            features.append("BERT_VECTOR")
        else:
            features = FEATURES

        print("- Features", features)
        print("- TIME_INTERVAL", EPOCHS, "/ BATCH_SIZE", BATCH_SIZE)
        print("- Epochs", EPOCHS, "/ LEARNING_RATE", LEARNING_RATE, "/ WEIGHT_DECAY", WEIGHT_DECAY)
        print("- LSTM Details: ", "hidden_dim", hidden_dim, "/ num_layers", num_layers, "/ dropout", dropout)

        print("STARTING TRAINING", features)
        opt.train(train_loader, val_loader, n_epochs=EPOCHS, test_loader_one=test_loader_one,
                  X_test=X_test,X_val = X_val,scaler=scaler)
