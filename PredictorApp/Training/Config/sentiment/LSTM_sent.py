TIME_INTERVALS = [
    # "10Min", "30Min", "1H", "2H",
    # "3H", "6H", "12H",
    "24H"]
BATCH_SIZE = 64
BERT_ENABLE = False
EPOCHS = 1000

TIME_DIFFERENCE_ENABLE = False

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
MAX_NO_IMPROVEMENTS = 15
WINDOW_SIZES = [
    2,
    5,
    10,
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
LOSS_FUNCTION = "MSE"
MODEL_NAME = "LSTM"
# MODEL_NAME = "CNNLSTM"
# MODEL_NAME = "DenseLSTM"

BERT_SIZE = 768
FEATURES = [
    # 'Volume',
    # 'tweet_vol',
    'sent_neg',
    'sent_neu',
    'sent_pos',
    'Close',
    # "High",
    # "Low",

]
TO_PREDICT_LABELS = ["Close"]

TEST_RATIO = 0.05
VAL_RATIO = 0.1

HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.2