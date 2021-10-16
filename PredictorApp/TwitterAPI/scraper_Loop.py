import datetime
from TwitterAPI.Keys import BEARER_TOKEN
from TweetScraper import TweetScraper
from datetime import timedelta


def scrape_tweets(start_time,end_time):

    scraper = TweetScraper(BEARER_TOKEN)

    data = scraper.get_tweets(start_time=start_time, end_time=end_time, tags="bitcoin OR btc -is:retweet")

    data.to_csv(path_or_buf="raw_data/scraped/tweets_" + str(start_time.date()) + str(end_time.date()) + ".csv")


def scrape_tweets_mixfill(start_time, end_time):
    scraper = TweetScraper(BEARER_TOKEN)
    data = scraper.get_tweets(start_time=start_time, end_time=end_time, tags="bitcoin lang:en -is:retweet")

    data.to_csv(path_or_buf="../raw_data/scraped/tweets_" + str(start_time.date()) + str(end_time.date()) + ".csv")


if __name__ == '__main__':
    # start_time = datetime.datetime(2021, 10, 10, 0, 0, 0, 0, datetime.timezone.utc)
    # end_time = datetime.datetime(2021, 10, 11, 0, 0, 0, 0, datetime.timezone.utc)
    # for i in range(14):
    #     print("start_time", start_time)
    #     print("end_time", end_time)
    #     scrape_tweets(
    #         start_time=start_time,
    #         end_time=end_time
    #     )
    #     start_time = start_time - timedelta(days=1)
    #     end_time = end_time - timedelta(days=1)

    start_time = datetime.datetime(2021, 9, 26, 0, 0, 0, 0, datetime.timezone.utc)
    end_time = datetime.datetime(2021, 9, 27, 0, 0, 0, 0, datetime.timezone.utc)
    for i in range(30):
        print("start_time", start_time)
        print("end_time", end_time)
        scrape_tweets(
            start_time=start_time,
            end_time=end_time
        )
        start_time = start_time - timedelta(days=1)
        end_time = end_time - timedelta(days=1)
