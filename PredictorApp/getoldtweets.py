import GetOldTweets3 as got

tweetCriteria = got.manager.TweetCriteria().setQuerySearch('europe')\
                                           .setSince("2015-05-01")\
                                           .setUntil("2015-09-30")\
                                           .setMaxTweets(1)
tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]
print(tweet.text)