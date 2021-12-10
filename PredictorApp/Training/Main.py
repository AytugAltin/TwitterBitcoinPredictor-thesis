from sklearn.preprocessing import MinMaxScaler
from Training.Trainer import Trainer
from Training.HelpersFunctions import *

import sys


def load_data(time_interval):
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

    compound_df = load_compound_data(time_interval, files)
    dataframes.append(compound_df)

    volume_df = load_volume_data(time_interval, start_date, end_date)
    dataframes.append(volume_df)

    bitcoin_df = load_bitcoin_data(time_interval, start_date, end_date)
    dataframes.append(bitcoin_df)

    sentiment_df = load_sentiment_volume_data(time_interval, files)
    dataframes.append(sentiment_df)

    df_raw = pd.DataFrame()
    for df in dataframes:
        df_raw = pd.concat([df_raw, df], axis=1)

    df_raw = df_raw.set_index('Date')
    print("DATA READY")

    return df_raw


def train_main(df_final_results, TIME_INTERVALS, window_size, model, optimizer,loss,
               time_difference_enabled, FEATURES, BERT_ENABLE, BERT_SIZE,
               SETS, BATCH_SIZE, MAX_NO_IMPROVEMENTS,EPOCHS):

    for time_interval in TIME_INTERVALS:
        try:
            df_raw = load_data(time_interval)
            zero_value = 0

            if time_difference_enabled:
                print("TIME DIFFERENCE ENABLED")
                zero_value = df_raw["Close"][window_size - 1]
                df_raw = time_differencing(df_raw)

            df_result = feature_extration(df_raw, (window_size - 1), FEATURES, BERT_ENABLE, BERT_SIZE)

            df_result["Close_next"] = df_result["Close"].shift(-1)
            df_result = df_result.iloc[:-1]
            to_predict_labels = ["Close_next"]

            print(" CROSS VALIDATION AND SPLITTING DATASET")
            sets = cross_validation_sets(df_result)

            for (training_set, validation_set, test_set, set_index) in sets:
                if set_index in SETS:
                    # region SetupLoaders

                    X_train, y_train = feature_label_split(training_set, to_predict_labels)
                    X_val, y_val = feature_label_split(validation_set, to_predict_labels)
                    X_test, y_test = feature_label_split(test_set, to_predict_labels)
                    scaler = MinMaxScaler()
                    train_loader, val_loader, test_loader, test_loader_one = \
                        create_data_loaders(X_train, X_val, y_train, y_val, X_test, y_test, scaler, BATCH_SIZE)

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

                    print("- TIME_INTERVAL", time_interval, "/ SET", set_index,
                          "/ WINDOW SIZE", window_size, "/ BATCH_SIZE", BATCH_SIZE)

                    print("- Training shape", training_set.shape,
                          "Validation shape", validation_set.shape,
                          "Testing shape", test_set.shape)

                    print("- Epochs", EPOCHS, "FEATURES", features)

                    print("OPTIMIZER",optimizer)
                    print("MODEL",model)
                    # endregion

                    print("STARTING TRAINING")

                    trainer = Trainer(model=model, loss_fn=loss, optimizer=optimizer, num_features=model.input_dim,
                                      name=name,root=root, max_no_improvements=MAX_NO_IMPROVEMENTS)

                    trainer.train(train_loader, val_loader, n_epochs=EPOCHS, test_loader_one=test_loader_one,
                                  X_test=X_test, X_val=X_val, X_train=X_train, scaler=scaler)

                    print("MODEL DONE TRAINING")
                    df_r = final_results(trainer=trainer, train_loader=train_loader, val_loader=val_loader,
                                         test_loader=test_loader_one,
                                         X_test=X_test, X_val=X_val, X_train=X_train, scaler=scaler,
                                         root=root, time_interval=time_interval, window_size=window_size,
                                         set_index=set_index, zero_value=zero_value,
                                         TIME_DIFFERENCE_ENABLE=time_difference_enabled,
                                         LEARNING_RATE=optimizer.defaults["lr"],
                                         WEIGHT_DECAY=optimizer.defaults["weight_decay"],
                                         BATCH_SIZE=BATCH_SIZE,FEATURES=FEATURES, BERT_ENABLE=BERT_ENABLE,
                                         MODEL_NAME=model.name)

                    df_final_results = df_final_results.append(df_r)
        except Exception as e:
            print(e)
            raise e

    return df_final_results

        # endregion
