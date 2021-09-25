import unicodedata
import pandas as pd
import contractions as contractions
import re
import nltk
import string
from nltk.corpus import stopwords
import regex as re
from TweetBert import TweetBert
from BotClassifier import BotClassifier
from Vader import Vader
import numpy


class FilterMechanism:

    def __init__(self):
        nltk.download('stopwords')
        self.tweet_bert = TweetBert()
        self.bot_classifier = BotClassifier()
        self.Vader = Vader()

    def pre_processing(self, tweet):
        tweet = tweet.replace('\n', '')
        tweet = tweet.replace('\t', '')
        tweet = tweet.replace('\r', '')
        return tweet

    def process(self, tweet):
        # Convert to lower case
        text = tweet.lower()

        # Convert www.* or https?://* to URL
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)

        # Remove @username
        text = re.sub('@[^\s]+', ' ', text)

        # Remove additional white spaces
        text = re.sub('[\s]+', ' ', text)

        # Replace #word with word
        text = re.sub(r'#([^\s]+)', r'\1', text)

        # trim
        text = text.strip('\'"')

        # remove punctuation
        for punctuation in string.punctuation:
            text = text.replace(punctuation, ' ')

        # words only
        text = ''.join([i for i in text if not i.isdigit()])

        # tokenize
        words = nltk.tokenize.casual_tokenize(text)

        # Remove stopwords
        stops = set(stopwords.words('english'))
        clean = [w for w in words if not w in stops]

        return ' '.join(clean)

    def get_url(self, x):
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x)
        if len(urls) == 0:
            return ' '
        else:
            return urls

    def find_tags(self, text):
        hashtag = set(part[1:] for part in text.split() if part.startswith('#'))

        if len(hashtag) == 0:
            hashtag = ' '

        return [''.join(map(str, l)) for l in hashtag]

    def find_promote(self, text):
        promoting = re.findall('subscribe | follow | promo', text)

        if len(promoting) == 0:
            return ' '
        else:
            return promoting

    def normalize_text(self, text):
        # remove non english alphabet letters
        return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")

    def remove_contractions(self, text):
        # can't -> can not
        return ' '.join([contractions.fix(word) for word in text.split()])

    def filter_pipeline(self, index, row, data_set):
        tweet = row["text"]

        tweet = self.pre_processing(tweet)

        vec = self.tweet_bert.tweet_to_vec_string(tweet)
        data_set._set_value(index, 'bert', vec)

        tweet = self.normalize_text(tweet)
        tweet = self.remove_contractions(tweet)

        tags = self.find_tags(tweet)
        data_set._set_value(index, 'tags', tags)

        promote = self.find_promote(tweet)
        data_set._set_value(index, 'promote', promote)

        tweet = self.process(tweet)
        data_set._set_value(index, 'filtered_text', tweet)

        if self.bot_classifier.tweet_is_bot(tweet):
            data_set._set_value(index, 'bot', True)
        else:
            data_set._set_value(index, 'bot', False)

        sentiment = self.Vader.get_score(tweet)
        data_set._set_value(index, 'sent_neg', sentiment["neg"])
        data_set._set_value(index, 'sent_neu', sentiment["neu"])
        data_set._set_value(index, 'sent_pos', sentiment["pos"])
        data_set._set_value(index, 'sent_compound', sentiment["compound"])


if __name__ == '__main__':

    filer_mechanism = FilterMechanism()

    data = pd.read_csv("data/2021/Bitcoin_tweets_en.csv", lineterminator='\n')
    data.dropna(how='any', inplace=True)

    data["bert"] = [[]] * len(data)

    for index, row in data.iterrows():
        try:
            filer_mechanism.filter_pipeline(index, row, data)

        except IndexError as err:
            print(err)

        except Exception as err:
            print(row["text"])
            raise err

    print("done")

    data.to_csv(path_or_buf="data/2021/Bitcoin_tweets_en_filtered.csv")
