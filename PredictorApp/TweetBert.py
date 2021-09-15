from transformers import AutoTokenizer
from TweetNormalizer import normalizeTweet
import torch


class TweetBert:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

    def tweet_to_vec_string(self, tweet):
        vec = self.tweet_to_vec(tweet)
        return str(torch.flatten(vec).tolist())

    def tweet_to_vec(self, tweet):
        line = normalizeTweet(tweet)
        input_ids = torch.tensor([self.tokenizer.encode(line)])
        return input_ids


