#Read the file using pandas and preprocess it
# get all the imports
from datetime import datetime
import os
import pandas as pd
from pandas import concat, DataFrame
from pandas import concat
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from numpy import  array

final_data_path = '../stock_datasets/final_data/'
fig = plt.figure(figsize=(15, 8), dpi= 80, facecolor='w', edgecolor='k')
csv_data = pd.read_csv(final_data_path + 'pre_stock2_data.csv')
row_date = csv_data['Date']
Open = csv_data['Date']
High= csv_data['High']
Low = csv_data['Low']
Close =  csv_data['Close']
news = csv_data.iloc[0:, 3:7].values
adj_close = csv_data.iloc[0:, 5:6].values
index = [pd.to_datetime(date, format='%Y-%m-%d').date() for date in row_date]
stock = (adj_close - np.min(adj_close)) / (np.max(adj_close) - np.min(adj_close))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax1.plot(index[0:],adj_close, label= 'adj_close')
plt.legend()
ax2.plot(index[0:],stock, label= 'Normalized_adj_close')
# Normalize the stock data
plt.show()