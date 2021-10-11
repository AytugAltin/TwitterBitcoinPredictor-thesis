import datetime
from TwitterAPI.Keys import BEARER_TOKEN
from TweetScraper import TweetScraper


def scrape_tweets():
    start_time = datetime.datetime(2018, 4, 1, 0, 0, 0, 0, datetime.timezone.utc)
    end_time = datetime.datetime(2018, 4, 2, 0, 0, 0, 0, datetime.timezone.utc)

    scraper = TweetScraper(BEARER_TOKEN)

    data = scraper.get_tweets(start_time=start_time, end_time=end_time, tags="bitcoin OR btc -is:retweet")

    data.to_csv(path_or_buf="raw_data/scraped/tweets_" + str(start_time.date()) + str(end_time.date()) + ".csv")


def scrape_tweets_mixfill(start_time, end_time):
    scraper = TweetScraper(BEARER_TOKEN)
    data = scraper.get_tweets(start_time=start_time, end_time=end_time, tags="bitcoin lang:en -is:retweet")

    data.to_csv(path_or_buf="raw_data/scraped/tweets_" + str(start_time.date()) + str(end_time.date()) + ".csv")


if __name__ == '__main__':
    scrape_tweets_mixfill(
        start_time=datetime.datetime(2018, 10, 10, 0, 0, 0, 0, datetime.timezone.utc),
        end_time=datetime.datetime(2018, 10, 11, 0, 0, 0, 0, datetime.timezone.utc)
    )

    scrape_tweets_mixfill(
        start_time=datetime.datetime(2018, 10, 11, 0, 0, 0, 0, datetime.timezone.utc),
        end_time=datetime.datetime(2018, 10, 12, 0, 0, 0, 0, datetime.timezone.utc)
    )

    scrape_tweets_mixfill(
        start_time=datetime.datetime(2018, 10, 1, 0, 0, 0, 0, datetime.timezone.utc),
        end_time=datetime.datetime(2018, 10, 2, 0, 0, 0, 0, datetime.timezone.utc)
    )

    scrape_tweets_mixfill(
        start_time=datetime.datetime(2018, 10, 22, 0, 0, 0, 0, datetime.timezone.utc),
        end_time=datetime.datetime(2018, 10, 23, 0, 0, 0, 0, datetime.timezone.utc)
    )

    scrape_tweets_mixfill(
        start_time=datetime.datetime(2018, 10, 28, 0, 0, 0, 0, datetime.timezone.utc),
        end_time=datetime.datetime(2018, 10, 29, 0, 0, 0, 0, datetime.timezone.utc)
    )


