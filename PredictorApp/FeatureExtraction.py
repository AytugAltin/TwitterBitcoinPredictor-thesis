from Dataset import *
import ast


def feature_label_split(df, target_col):
    # from https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b
    y = df[target_col]
    X = df.drop(columns=target_col)
    return X, y


def train_val_test_split(df, target_col, test_ratio):
    # from https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test


def generate_time_lags(df, n_lags, features):
    for feature in features:
        df = generate_time_lag(df, n_lags, feature)
    df = df.iloc[n_lags:]
    return df


def generate_time_lag(df, n_lags, feature):
    # from https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"{feature + '_l' + str(n)}"] = df_n[feature].shift(n)
    return df_n


def time_lag_features(df, features, N):
    df = df[features]
    df_generated = generate_time_lags(df, N, features)

    return df_generated


def create_bert_feature(df_bert, window_size):
    return 0


def time_difference(df, tag):
    df[tag] = df[tag].diff()
    df.at[df.index[0], tag] = 0
    return df
