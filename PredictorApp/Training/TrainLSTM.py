import datetime
import os

import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from DataProcessing.DataGrouper import group_tweetsdata, group_volumedata, group_bitcoindata
from Device import DEVICE
from FeatureExtraction import *
from Models.Hybrid.DenseLSTM import DenseLSTM
from Models.LSTM.LSTM import LSTM
from Models.Hybrid.CNNLSTM import CNNLSTM

from Trainer import Trainer

# region CONFIG
TIME_INTERVAL = "3H"
BATCH_SIZE = 64
BERT_ENABLE = False
EPOCHS = 10000

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
TO_PREDICT_LABEL = "Close"


# endregion

# region Helpers
def load_tweet_data(time_interval, files, renew=False):
    print("READING TWEETS")
    potential_path = "../Data/2018-Weighted/cache/" + "TWEETS" + "(" + time_interval + ")" + ".csv"

    if os.path.isfile(potential_path) and not renew:
        # Read CACHED FILE
        print(" - Cached File Found")
        df = pd.read_csv(potential_path)
        print(" - LOADED", potential_path)
        return df

    print(" - NO Cached File Found")

    root_path = "../Data/2018-Weighted/grouped/"
    tweets_data = pd.DataFrame()
    for file_path in files:
        print("     - reading", file_path)
        temp = pd.read_csv(root_path + file_path)
        tweets_data = tweets_data.append(temp)

    print(" - Grouping")
    grouped_df = group_tweetsdata(tweets_data, time_interval)
    print(" - Writing")
    grouped_df.to_csv(potential_path)
    print(" - Tweets Data Cached in", potential_path)

    print(" - Fixing BERT Features")
    final_df = fix_bert(potential_path)
    print(" - BERT Features Fixed", potential_path)

    print(" - Writing With BERT features")
    final_df.to_csv(potential_path)

    print(" ! Reading Tweets Data DONE...")
    return final_df


def fix_bert(path):
    bert = ""
    for i in range(0, 768):
        bert += ",bert" + str(i)

    bert += ","
    data = None
    with open(path, 'r') as infile:
        data = infile.read()
        data = data.replace('"', '')
        data = data.replace('[', '')
        data = data.replace(']', '')
        data = data.replace(',bert,', bert)
        infile.close()

    with open(path, 'w') as outfile:
        outfile.write(data)

    grouped_df = pd.read_csv(path)
    return grouped_df


def load_volume_data(time_interval, start_date, end_date, renew=False):
    print("READING VOLUME")
    potential_path = "../Data/2018-Weighted/cache/" + "VOLUME" + "(" + time_interval + ")" + ".csv"

    if os.path.isfile(potential_path) and not renew:
        # Read CACHED FILE
        print(" - Cached File Found")
        df = pd.read_csv(potential_path)
        return df
    print(" - Cached File NOT Found")

    file_path = "../Data/2018tweets/2018(03-08--03-11).csv"
    print("     - reading", file_path)
    volume_data = pd.read_csv(file_path)

    print(" - Grouping")
    grouped_df = group_volumedata(volume_data, time_interval)

    print("     - Filtering Between", start_date.date(), "and", end_date.date())
    grouped_df = grouped_df.loc[(grouped_df['date'] >= start_date.replace(tzinfo=None))
                                & (grouped_df['date'] < end_date.replace(tzinfo=None))]

    print(" - Writing")
    grouped_df.to_csv(potential_path)
    print(" - Volume Data Cached in", potential_path)

    print(" ! Reading Volume Data DONE...")
    return grouped_df


def load_bitcoin_data(time_interval, start_date, end_date, renew=False):
    print("READING BITCOIN")
    potential_path = "../Data/2018-Weighted/cache/" + "BITCOIN" + "(" + time_interval + ")" + ".csv"

    if os.path.isfile(potential_path) and not renew:
        # Read CACHED FILE
        print(" - Cached File Found")
        bitcoin_df = pd.read_csv(potential_path)
        return bitcoin_df
    print(" - Cached File NOT Found")

    file_path = "../Data/bitcoin/Bitstamp_BTCUSD_2018_minute.csv"
    print("     - reading", file_path)
    bitcoin_data = pd.read_csv(file_path)

    print("     - Filtering Between", start_date.date(), "and", end_date.date())
    # bitcoin_data = bitcoin_data.sort_values(by='Date')
    bitcoin_data['Date'] = pd.to_datetime(bitcoin_data['Date'])
    bitcoin_data = bitcoin_data.loc[(bitcoin_data['Date'] >= start_date.replace(tzinfo=None))
                                    & (bitcoin_data['Date'] < end_date.replace(tzinfo=None))]

    print(" - Grouping")
    grouped_df = group_bitcoindata(bitcoin_data, time_interval)

    print(" - Writing")
    grouped_df.to_csv(potential_path)
    print(" - Bitcoin Data Cached in", potential_path)

    print(" ! Reading Bitcoin Data DONE...")
    return grouped_df


# endregion

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

    print("DATA READY")
    # endregion

    # region Feature Extraction
    print("EXTRACTING FEATURES")
    df_result = time_difference(df_raw, "Close")
    df_result = time_lag_features(df_result, FEATURES, N=WINDOW_SIZE)
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

    print("SPLITTING DATASET")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_result, TO_PREDICT_LABEL, 0.05)

    # endregion

    # region Fixing loaders
    # from https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b
    scaler = MinMaxScaler()

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
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

    # endregion

    # region MODEL
    input_dim = len(X_train.columns)
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

    opt = Trainer(model=model, loss_fn=loss, optimizer=optimizer)

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
    opt.train(train_loader, val_loader, batch_size=BATCH_SIZE, n_epochs=EPOCHS,
              n_features=input_dim, test_loader_one=test_loader_one, X_test=X_test,
              scaler=scaler)
