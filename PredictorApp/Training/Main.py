from time import strftime
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from Models.Models import *
from Models.LSTM.LSTMModel import LSTMModel
from Training.Trainer import Trainer
from HelpersFunctions import *
from Training.LossFunctions import *
import sys

# region CONFIG
CONFIG = sys.argv[1]

if CONFIG == "LSTM_BASELINE":
    from Training.Config.baseline.LSTM_baseline import *
elif CONFIG == "LSTM_BASELINE_TD":
    from Training.Config.baseline.LSTM_baselineTD import *
elif CONFIG == "LSTM_sent":
    from Training.Config.sentiment.LSTM_sent import *
elif CONFIG == "EXP1":
    from Training.Config.EXPERIMENTAL import *
elif CONFIG == "EXP2":
    from Training.Config.EXPERIMENTAL2 import *
elif CONFIG == "RNN":
    from Training.Config.RNN import *
elif CONFIG == "DENSE":
    from Training.Config.DENSELSTM_CFG import *

print(CONFIG, "LOADED")

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


    # endregion

    zero_value = 0
    if TIME_DIFFERENCE_ENABLE:
        print("TIME DIFFERENCE ENABLED")
        zero_value = df_raw["Close"][window_size - 1]
        df_raw = time_differencing(df_raw)

    df_result = feature_extration(df_raw, window_size, FEATURES, BERT_ENABLE, BERT_SIZE)

    df_result["Close_next"] = df_result["Close"].shift(-1)
    df_result = df_result.iloc[:-1]
    TO_PREDICT_LABELS = ["Close_next"]

    print(" CROSS VALIDATION AND SPLITTING DATASET")
    sets = cross_validation_sets(df_result)
    # sets.reverse()
    num_features = len(df_result.columns) - len(TO_PREDICT_LABELS)
    try:
        for (training_set, validation_set, test_set, set_index) in sets:
            if set_index in SETS:
                # region SetupLoaders

                X_train, y_train = feature_label_split(training_set, TO_PREDICT_LABELS)
                X_val, y_val = feature_label_split(validation_set, TO_PREDICT_LABELS)
                X_test, y_test = feature_label_split(test_set, TO_PREDICT_LABELS)
                scaler = MinMaxScaler()
                train_loader, val_loader, test_loader, test_loader_one = \
                    create_data_loaders(X_train, X_val, y_train, y_val, X_test, y_test, scaler, BATCH_SIZE)

                # endregion

                # region MODEL
                input_dim = num_features
                hidden_dim = HIDDEN_DIM
                num_layers = NUM_LAYERS
                output_dim = 1
                dropout = DROPOUT

                Models = {}
                try:
                    Models["LSTM"] = LSTMModel(input_dim, hidden_dim, num_layers, output_dim,
                                               dropout, BATCH_SIZE).to(DEVICE)
                    Models["RNN"] = RNNModel(input_dim, hidden_dim, num_layers, output_dim,
                                          dropout, BATCH_SIZE).to(DEVICE)
                    Models["EXPMODEL"] = EXPMODEL(input_dim, hidden_dim, num_layers, output_dim,
                                                dropout, BATCH_SIZE).to(DEVICE)
                    Models["EXPMODEL2"] = EXPMODEL2(input_dim, hidden_dim, num_layers, output_dim,
                                                dropout, BATCH_SIZE).to(DEVICE)
                    Models["DenseLSTM"] = DenseLSTM(input_dim, hidden_dim, num_layers, output_dim,
                                                    dropout, BATCH_SIZE, BERT_SIZE).to(DEVICE)
                    Models["CNNLSTM"] = CNNLSTM(input_dim, hidden_dim, num_layers, output_dim,
                                                dropout, BATCH_SIZE, BERT_SIZE, window_size).to(DEVICE)
                except Exception as e:
                    print(e)


                model = Models[MODEL_NAME]

                print(model)

                # endregion

                # region Root_NAME
                root = "../Logs/"
                create_folder(root)
                root = root + "TI(" + str(time_interval) + ")/"
                create_folder(root)
                root = root + "WS(" + str(window_size) + ")/"
                create_folder(root)
                root = root + "SET" + str(set_index) + "/"
                create_folder(root)
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
                print("- TIME_INTERVAL", time_interval, "/ SET", set_index,
                      "/ WINDOW SIZE", window_size, "/ BATCH_SIZE",BATCH_SIZE)

                print("- Training shape", training_set.shape,
                      "Validation shape", validation_set.shape,
                      "Testing shape", test_set.shape)

                print("- Epochs", EPOCHS, "/ LEARNING_RATE", LEARNING_RATE,
                      "/ WEIGHT_DECAY", WEIGHT_DECAY)
                print("- LSTM Details: ", "input_dim", input_dim, "hidden_dim",
                      hidden_dim, "/ num_layers", num_layers, "/ dropout", dropout)
                print("FEATURES", features)
                # endregion

                print("STARTING TRAINING")
                loss = LOSS_FUNCTIONS[LOSS_FUNCTION]
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                trainer = Trainer(model=model, loss_fn=loss, optimizer=optimizer, num_features=input_dim, name=name,
                                  root=root, max_no_improvements=MAX_NO_IMPROVEMENTS)

                trainer.train(train_loader, val_loader, n_epochs=EPOCHS, test_loader_one=test_loader_one,
                              X_test=X_test, X_val=X_val, X_train=X_train, scaler=scaler)

                print("MODEL DONE TRAINING")
                df_r = final_results(trainer=trainer, train_loader=train_loader, val_loader=val_loader,
                                     test_loader=test_loader_one,
                                     X_test=X_test, X_val=X_val, X_train=X_train, scaler=scaler,
                                     root=root, time_interval=time_interval, window_size=window_size,
                                     set_index=set_index,zero_value=zero_value,
                                     TIME_DIFFERENCE_ENABLE=TIME_DIFFERENCE_ENABLE,
                                     LEARNING_RATE=LEARNING_RATE, WEIGHT_DECAY=WEIGHT_DECAY, BATCH_SIZE=BATCH_SIZE,
                                     FEATURES=FEATURES, BERT_ENABLE=BERT_ENABLE, MODEL_NAME=MODEL_NAME)

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
