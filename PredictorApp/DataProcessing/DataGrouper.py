import ast
import pandas as pd


def meanlist(listoflists):
    divider = len(listoflists)
    try:
        summed = listoflists.pop()
        while len(listoflists) > 0:
            summed = [sum(item) for item in zip(summed, listoflists.pop())]
        mean = []
        for number in summed:
            mean.append(number / divider)
        return mean
    except:
        return []


def convert_to_datetime(data_frame, key):
    try:
        data_frame[key] = pd.to_datetime(data_frame[key], format='%d/%m/%Y %H:%M:%S', utc=True)
    except:
        try:
            data_frame[key] = pd.to_datetime(data_frame[key], format='%d/%m/%Y %H:%M', utc=True)
        except:
            data_frame[key] = pd.to_datetime(data_frame[key], format='%Y-%m-%d %H:%M:%S', utc=True)

    return data_frame


def group_tweetsdata(tweets, time_interval, bot_filtering=True):
    data = tweets

    data = data.sort_values(by='date')
    data = data[["date", "bert", "sent_neg", "sent_neu", "sent_pos", "bot"]]

    data = data[data.bot != bot_filtering][["date", "bert", "sent_neg", "sent_neu", "sent_pos"]]

    data = convert_to_datetime(data, "date")
    data['date'] = data["date"].dt.tz_localize(None)
    data = data.set_index("date")

    aggregations = {
        'sent_neg': 'mean',
        'sent_neu': 'mean',
        'sent_pos': 'mean',
        "bert": lambda x: meanlist([ast.literal_eval(y) for y in x.values]),
        "count": 'sum'
    }
    data["count"] = 1
    grouped_data = data.groupby(pd.Grouper(freq=time_interval)).agg(aggregations)

    return grouped_data


def group_volumedata(volume_data, time_interval):
    data = volume_data
    data = data.sort_values(by='date')
    data = data[["date"]]

    data = convert_to_datetime(data, "date")
    data['date'] = data["date"].dt.tz_localize(None)
    data = data.set_index("date")

    grouped_data = data.groupby(pd.Grouper(freq=time_interval)).size().reset_index(name='tweet_vol')
    return grouped_data
