from datetime import timedelta

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from Dataset import *

INPUT_FEATURES = [
    'Volume',
    'tweet_vol',
    'sent_neg', 'sent_neu', 'sent_pos',
    # 'bert',
    # 'count'
]
OUTPUT_FEATURES = ['Close']


class Model:
    def __init__(self, learning_rate, num_layers, size, size_layer, forget_bias=0.8):
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple=False)

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size_layer) for _ in range(num_layers)],
                                                state_is_tuple=False)

        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, size))

        drop = tf.contrib.rnn.DropoutWrapper(rnn_cells, output_keep_prob=forget_bias)

        self.hidden_layer = tf.placeholder(tf.float32, (None, num_layers * 2 * size_layer))

        self.outputs, self.last_state = tf.nn.dynamic_rnn(drop, self.X,
                                                          initial_state=self.hidden_layer,
                                                          dtype=tf.float32)

        self.logits = tf.layers.dense(self.outputs[-1], size,
                                      kernel_initializer=tf.glorot_uniform_initializer())

        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(self.cost)



def predict_future(future_count, df, dates, indices={}):
    date_ori = dates[:]
    cp_df = df.copy()
    output_predict = np.zeros((cp_df.shape[0] + future_count, cp_df.shape[1]))
    output_predict[0, :] = cp_df.iloc[0]
    upper_b = (cp_df.shape[0] // timestamp) * timestamp
    init_value = np.zeros((1, num_layers * 2 * size_layer))
    for k in range(0, (df.shape[0] // timestamp) * timestamp, timestamp):
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(
                    cp_df.iloc[k : k + timestamp], axis = 0
                ),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[k + 1 : k + timestamp + 1] = out_logits
    out_logits, last_state = sess.run(
        [modelnn.logits, modelnn.last_state],
        feed_dict = {
            modelnn.X: np.expand_dims(cp_df.iloc[upper_b:], axis = 0),
            modelnn.hidden_layer: init_value,
        },
    )
    init_value = last_state
    output_predict[upper_b + 1 : cp_df.shape[0] + 1] = out_logits
    cp_df.loc[cp_df.shape[0]] = out_logits[-1]
    date_ori.append(date_ori[-1] + timedelta(hours = 1))
    if indices:
        for key, item in indices.items():
            cp_df.iloc[-1,key] = item
    for i in range(future_count - 1):
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(cp_df.iloc[-timestamp:], axis = 0),
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[cp_df.shape[0], :] = out_logits[-1, :]
        cp_df.loc[cp_df.shape[0]] = out_logits[-1, :]
        date_ori.append(date_ori[-1] + timedelta(hours = 1))
        if indices:
            for key, item in indices.items():
                cp_df.iloc[-1,key] = item
    return {'date_ori': date_ori, 'df': cp_df.values}

def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer

if __name__ == '__main__':
    dataset = CombinedDataset(csv_file="Data/2018tweets/Objects/(60Min).csv",
                              input_features=INPUT_FEATURES, output_features=OUTPUT_FEATURES)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(dataset.data, INPUT_FEATURES,
                                                                          OUTPUT_FEATURES, 0.2)

    df = dataset.data

    minmax = MinMaxScaler().fit(df[["sent_neg", "sent_neu", "sent_pos", 'tweet_vol', 'Close']].astype('float32'))
    df_scaled = minmax.transform(df[["sent_neg", "sent_neu", "sent_pos", 'tweet_vol', 'Close']].astype('float32'))
    # TODO min max scale voor sentiment waardes een, goed idee?
    df_scaled = pd.DataFrame(df_scaled)

    num_layers = 1
    learning_rate = 0.005
    size_layer = 128
    timestamp = 5
    epoch = 500
    dropout_rate = 0.6

    dates = pd.to_datetime(df.index).tolist()

    modelnn = Model(learning_rate, num_layers, df_scaled.shape[1], size_layer, dropout_rate)
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())


    for i in range(epoch):
        init_value = np.zeros((1, num_layers * 2 * size_layer))
        total_loss = 0
        for k in range(0, (df_scaled.shape[0] // timestamp) * timestamp, timestamp):
            batch_x = np.expand_dims(df_scaled.iloc[k: k + timestamp].values, axis=0)
            batch_y = df_scaled.iloc[k + 1: k + timestamp + 1].values
            last_state, _, loss = sess.run([modelnn.last_state,
                                            modelnn.optimizer,
                                            modelnn.cost], feed_dict={modelnn.X: batch_x,
                                                                      modelnn.Y: batch_y,
                                                                      modelnn.hidden_layer: init_value})
            init_value = last_state
            total_loss += loss
        total_loss /= (df.shape[0] // timestamp)
        if (i + 1) % 100 == 0:
            print('epoch:', i + 1, 'avg loss:', total_loss)

    predict_30 = predict_future(30, df_scaled, dates)
    predict_30['df'] = minmax.inverse_transform(predict_30['df'])

    signal = np.copy(df['Close'].values)

    plt.figure(figsize=(15, 7))
    plt.plot(np.arange(len(predict_30['date_ori'])), anchor(predict_30['df'][:, -1], 0.5), label='predict signal')
    plt.plot(np.arange(len(signal)), signal, label='real signal')
    plt.legend()
    plt.show()