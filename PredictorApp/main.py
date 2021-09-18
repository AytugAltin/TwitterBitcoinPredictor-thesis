from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from BotClassifier import BotClassifier
from Datasets import DatasetCreator


def sentiment_scores(sentence):
    print("--", sentence, "--")
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)

    print("Overall sentiment dictionary is : ", sentiment_dict)
    print("sentence was rated as ", sentiment_dict['neg'] * 100, "% Negative")
    print("sentence was rated as ", sentiment_dict['neu'] * 100, "% Neutral")
    print("sentence was rated as ", sentiment_dict['pos'] * 100, "% Positive")

    print("Sentence Overall Rated As", end=" ")

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05:
        print("Positive")
    elif sentiment_dict['compound'] <= - 0.05:
        print("Negative")
    else:
        print("Neutral")


def sentiment_testing():
    sentences = [
        "This was a not good movie.",
    ]
    for sentence in sentences:
        sentiment_scores(sentence)
        print("")


def bot_testing():
    bot_clsfr = BotClassifier()

    print(bot_clsfr.tweet_is_bot("start trading bitcoin"))


def create_dataset():
    dataset_creator = \
        DatasetCreator(
            tweets_path="data/TweetsBTC_16mil/filtered/16_en_filtered.csv",
            volume_path="data/TweetsBTC_16mil/peryear/16.csv",
            bitcoin_path="data/bitcoin/gemini_BTCUSD_2017_1min.csv",
            time_interval="30Min"
        )
    print()


if __name__ == '__main__':
    # sentiment_testing()

    # bot_testing()

    create_dataset()
