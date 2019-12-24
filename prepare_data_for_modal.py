import pandas as pd
from pandas import concat, DataFrame
import numpy as np
from numpy import  array

from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X)

# raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# # choose a number of time steps
# n_steps = 3
# # split into samples
# X, y = split_sequence(raw_seq, n_steps)
# print(X)
# print(y)

def prepare_data(stock_number):
    #Read the file using pandas and preprocess it
    BASE_URL = './stock_datasets/'
    final_data_path = BASE_URL + 'final_data/'
    # read the csv file
    pre_stock1_data = pd.read_csv(final_data_path + 'pre_stock'+stock_number+'_data.csv')
    # process the colums of the file
    # reaname the column
    pre_stock1_data = pre_stock1_data.rename(columns={'Adj Close': 'Price'})
    # remove the unused column
    pre_stock1_data = pre_stock1_data.drop('Close', axis=1)
    # add a blank month column in the data field
    pre_stock1_data['Month'] = int
    # add the data into the month column, it will take time
    for i in range(len(pre_stock1_data)):
        pre_stock1_data['Month'][i] = int(pre_stock1_data['Date'][i].split('/')[1])

    # arrange the final sequence of column
    pre_stock1_data = pre_stock1_data[
        ['Open', 'High', 'Low', 'compound', 'neg', 'neu', 'pos', 'Month', 'Date', 'Price']]

    # Split the data into news and stock
    # only the Opening price of the stock is taken.
    stock = pre_stock1_data.iloc[:, :1].values
    news = pre_stock1_data.iloc[0:, 3:7].values
    # Normalize the stock data
    stock = (stock - np.min(stock)) / (np.max(stock) - np.min(stock))
    return pre_stock1_data, stock, news
