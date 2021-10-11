from transformers import AutoTokenizer, AutoModel

import torch

inputlength = 128


class BERTTweet:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False, normalization=True)
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base").to('cuda:0')

    def tweet_to_vec_string(self, tweet):
        vec = self.tweet_to_vec(tweet)
        return torch.flatten(vec).tolist()

    def tweet_to_vec(self, tweet):
        tokens = self.tokenizer.encode(tweet)

        if len(tokens) > inputlength:
            tokens = tokens[0:inputlength]

        input_ids = torch.tensor([tokens]).to('cuda:0')

        with torch.no_grad():
            features = self.bertweet(input_ids)

        return features.pooler_output
