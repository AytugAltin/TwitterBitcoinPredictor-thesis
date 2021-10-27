from Dataset import *
import ast
FEATURES = [
    'Volume',
    'tweet_vol',
    # 'sent_neg', 'sent_neu', 'sent_pos',
    'Close'
]


def generate_time_lags(df, n_lags, features):
    for feature in features:
        df = generate_time_lag(df, n_lags, feature)
    return df


def bert(df, dataset, enable=False):
    if not enable:
        return df

    list = []

    for i in range(0,768):
        list.append("bert"+str(i))

    df_bert = dataset.data[list]

    df = pd.concat([df, df_bert], axis=1)
    return df





def generate_time_lag(df, n_lags, feature):

    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"{feature + str(n)}"] = df_n[feature].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n


def extract_features(dataset):
    N = 100
    df = dataset.data[FEATURES]

    df = bert(df, dataset, enable=True)

    df_generated = generate_time_lags(df, N, FEATURES)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_generated, "Close", 0.2)

    return X_train, X_val, X_test, y_train, y_val, y_test
