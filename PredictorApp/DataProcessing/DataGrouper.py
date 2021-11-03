import ast
import pandas as pd


# region Helpers
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


def sum_list(listoflists):
    listoflists= listoflists.values.tolist()
    try:
        listoflists.remove("[]")
    except:
        pass
    try:
        summed = listoflists.pop()
        while len(listoflists) > 0:
            summed = [sum(item) for item in zip(summed, listoflists.pop())]
        return summed
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


def multiply_row(row, multiplyer):
    row["sent_neg"] = row["sent_neg"] * multiplyer
    row["sent_neu"] = row["sent_neu"] * multiplyer
    row["sent_pos"] = row["sent_pos"] * multiplyer

    try:
        row["bert"] = row["bert"].strip()
        row["bert"] = list(map(float, list(row["bert"].strip('[]').split(','))))
    except:
        pass

    try:
        row["bert"] = [element * multiplyer for element in row["bert"]]
    except:
        pass

    return row


def divide_row(row, divider):
    try:
        row["sent_neg"] = row["sent_neg"] / divider
        row["sent_neu"] = row["sent_neu"] / divider
        row["sent_pos"] = row["sent_pos"] / divider
    except:
        pass

    try:
        row["bert"] = row["bert"].strip()
        row["bert"] = list(map(float, list(row["bert"].strip('[]').split(','))))
    except:
        pass

    try:
        row["bert"] = [element / divider for element in row["bert"]]
    except:
        pass

    return row


# endregion


def group_tweetsdata(tweets, time_interval, bot_filtering=True):
    data = tweets
    data = data.sort_values(by='date')

    data = convert_to_datetime(data, "date")
    data['date'] = data["date"].dt.tz_localize(None)
    data = data.set_index("date")

    data = data.apply(lambda row: multiply_row(row, row["weight"]), axis=1)

    aggregations = {
        'sent_neg': 'sum',
        'sent_neu': 'sum',
        'sent_pos': 'sum',
        "bert": lambda x: sum_list(x),
        "count": 'sum',
        "weight": 'sum'
    }
    grouped_data = data.groupby(pd.Grouper(freq=time_interval)).agg(aggregations)

    grouped_data = grouped_data.apply(lambda row: divide_row(row, row["weight"]), axis=1)

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


def group_bitcoindata(bitcoin_data,time_interval):
    data = bitcoin_data
    data = data.sort_values(by='Date')
    data = data[["Date", "Open", "High", "Low", "Close", "Volume", "Volume USD"]]
    data = data.set_index("Date")
    data.index = pd.to_datetime(data.index)

    aggregations = {
        'Open': lambda x: x.iloc[0],
        'High': 'max',
        'Low': 'min',
        "Close": lambda x: x.iloc[-1],
        "Volume": 'sum'
    }

    grouped_data = data.groupby(pd.Grouper(freq=time_interval)).agg(aggregations)
    return grouped_data
