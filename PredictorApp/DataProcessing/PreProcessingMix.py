import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = "../raw_data/mix/Raw/18/PART 1/Feb 2018/2018-02-18.csv"

fields = ["date"]
volume_data = pd.read_csv(file_path, sep=";", usecols=fields)

volume_data = volume_data.sort_values(by='date')
volume_data['weight'] = 1

try:
    volume_data['date'] = pd.to_datetime(volume_data['date'], format='%d/%m/%Y %H:%M')
except:
    volume_data['date'] = pd.to_datetime(volume_data['date'], format='%Y-%m-%d %H:%M:%S')

volume_data = volume_data.set_index("date")

grouped_data = volume_data.groupby(pd.Grouper(freq="10Min")).size().reset_index(name='tweet_vol')
plt.close()
grouped_data.plot(x='date')
plt.show()

print("Done")