TIME_INTERVALS = [
    # "10Min",
    # "30Min",
    # "1H",
    # "2H",
    #  "3H",
    #  "6H",
    # "12H",
    "24H"
]
BATCH_SIZE = 8
BERT_ENABLE = False
EPOCHS = 1000

TIME_DIFFERENCE_ENABLE = False
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
MAX_NO_IMPROVEMENTS = 5

LOSS_FUNCTION = "MSE"

WINDOW_SIZES = [
    1,
    # 2,
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
    # 'sent_neg',
    # 'sent_neu',
    # 'sent_pos',
    'Close'
]
TO_PREDICT_LABELS = ["Close"]

TEST_RATIO = 0.05
VAL_RATIO = 0.1

HIDDEN_DIM = 32
NUM_LAYERS = 1
DROPOUT = 0.2
