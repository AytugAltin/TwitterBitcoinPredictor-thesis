# An in-depth analysis on the impact of Twitter features on bitcoin’s price prediction
# Abstract:

Influential people such as the famous entrepreneur Elon Musk have altered the price of several cryptocurrencies by simply tweeting about them. Different celebrities, state mayors, and senators have influenced cryptocurrencies using Twitter making the platform a source of information to predict cryptocurrency-related trends.

We introduce a variety of Twitter features to forecast bitcoin's price. Moreover, we introduce weighted tweet sentiment features that reflect the overall opinion about bitcoin on Twitter, tweet volume features indicating the overall activity and interest in bitcoin on Twitter and, a weighted BERT feature vector that encapsulates the information found in tweets regarding bitcoin. The prediction tasks include predicting the absolute price in bitcoin and the price difference. In particular, we have used long short-term memory (LSTM) based models that take a group of these features as input alongside the absolute price or price difference. 

As opposed to related research, we found that our Tweet sentiment features, obtained by the VADER sentiment model, did not produce an improvement in the prediction tasks for our data over our baseline models. They did, however, react to changes in bitcoin's price.

We introduce the first implementation of BERTTweet produced language vectors to represent the tweet information in bitcoin prediction tasks. Our results suggest that using these BERT vectors as an input to the prediction model seems to introduce noise into the data and even decreases the model's performance.

Furthermore, tweet volume appears to increase the correlation with bitcoin's price when bots and unavailable tweets are filtered from the dataset. It seems that removing redundant tweets improves the model in predicting upward and downward trends in bitcoin's price. Both a large increase and decrease in bitcoin's price seem to increase the volume of tweets regarding bitcoin on Twitter. This indicates that large fluctuations in bitcoin's price cause spikes in the number of tweets that are posted regarding bitcoin.

Our results show that by increasing the time intervals that the features represent, the correlation with bitcoin's price increases. Features representing larger time intervals have a higher chance of including data that has a higher correlation to bitcoin's price compared to features representing smaller time intervals.

# Twitter Data
Twitter data used in this thesis can be found in this repository under DATA/
