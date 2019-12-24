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


def show_all_adjclose():
    final_data_path = './final_data/'
    files = os.listdir(final_data_path)
    fig = plt.figure(figsize=(15, 8), dpi= 80, facecolor='w', edgecolor='k')
    for i in range(1,9):
        csv_data = pd.read_csv(final_data_path + 'pre_stock'+ str(i)+'_data.csv')
        row_date = csv_data['Date']
        news = csv_data.iloc[0:, 3:7].values
        adj_close = csv_data.iloc[0:, 5:6].values
        index = [pd.to_datetime(date, format='%Y-%m-%d').date() for date in row_date]
        plt.plot(index[0:] ,adj_close, label= 'stock_'+str(i))
    plt.legend()
    plt.show()


def show_stock1():
    final_data_path = './final_data/'
    fig = plt.figure(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
    csv_data = pd.read_csv(final_data_path + 'pre_stock1_data.csv')
    row_date = csv_data['Date']
    Open = csv_data['Date']
    High = csv_data['High']
    Low = csv_data['Low']
    Close = csv_data['Close']
    news = csv_data.iloc[0:, 3:7].values
    adj_close = csv_data.iloc[0:, 5:6].values
    index = [pd.to_datetime(date, format='%Y-%m-%d').date() for date in row_date]
    # plt.plot(index[0:] ,Open, label= 'Open')
    plt.plot(index[0:], High, label='High')
    plt.plot(index[0:], Low, label='Low')
    plt.plot(index[0:], Close, label='Close')
    plt.plot(index[0:], adj_close, label='adj_close')
    plt.legend()
    plt.show()

def show_Normalised_data():
    fig = plt.figure(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
    csv_data = pd.read_csv(final_data_path + 'pre_stock1_data.csv')
    row_date = csv_data['Date']
    Open = csv_data['Date']
    High = csv_data['High']
    Low = csv_data['Low']
    Close = csv_data['Close']
    news = csv_data.iloc[0:, 3:7].values
    adj_close = csv_data.iloc[0:, 5:6].values
    index = [pd.to_datetime(date, format='%Y-%m-%d').date() for date in row_date]
    stock = (adj_close - np.min(adj_close)) / (np.max(adj_close) - np.min(adj_close))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.plot(index[0:], adj_close, label='adj_close')
    ax1.legend()
    ax2.plot(index[0:], stock, label='Normalized_adj_close')
    ax2.legend()
    # Normalize the stock data
    plt.show()

def show_price_and_sensetive():
    fig = plt.figure(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
    csv_data = pd.read_csv(final_data_path + 'pre_stock1_data.csv')
    row_date = csv_data['Date']
    Open = csv_data['Date']
    compound = csv_data['compound']
    adj_close = csv_data.iloc[0:, 5:6].values
    index = [pd.to_datetime(date, format='%Y-%m-%d').date() for date in row_date]
    stock = (adj_close - np.min(adj_close)) / (np.max(adj_close) - np.min(adj_close))
    plt.plot(index[0:], compound, label='compound')
    plt.plot(index[0:], stock, label='Normalized_adj_close')
    plt.legend()
    # Normalize the stock data
    plt.show()