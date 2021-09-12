import unicodedata
import pandas as pd
import contractions as contractions
import re
import nltk
import string
from nltk.corpus import stopwords
import regex as re
import TweetBert


class FilterMechanism:

    def __init__(self):
        nltk.download('stopwords')
        self.tweet_bert = TweetBert()

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
        return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")

    def remove_contractions(self, text):
        return ' '.join([contractions.fix(word) for word in text.split()])

    def filter_pipeline(self, index, row, data_set):
        text = row["text"]
        vec = self.tweet_bert.tweet_to_vec(text)
        data_set._set_value(index, 'bert', vec)

        text = self.normalize_text(text)
        text = self.remove_contractions(text)

        tags = self.find_tags(text)
        data_set._set_value(index, 'tags', tags)

        promote = self.find_promote(text)
        data_set._set_value(index, 'promote', promote)

        text = self.process(text)
        data_set._set_value(index, 'filtered_text', text)


if __name__ == '__main__':

    filer_mechanism = FilterMechanism()
    sets = ["16_en", "17_en", "18_en", "19_en"]

    for setName in sets:
        data = pd.read_csv("data/TweetsBTC_16mil/english/" + setName + ".csv", lineterminator='\n')
        data.dropna(how='any', inplace=True)

        for index, row in data.iterrows():
            try:
                filer_mechanism.filter_pipeline(index, row, data)

            except IndexError as err:
                print(err)

            except Exception as err:
                print(err)
                print(row["text"])

        print(setName, "done")

        data.to_csv(path_or_buf="data/TweetsBTC_16mil/filtered/" + setName + "_filtered.csv")
