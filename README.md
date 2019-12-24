stock_predict
#This project is all about exploring the data of stocks and trying to predict the Adjusted Closing Price of the stocks by analysing the previous Opening price of stock and news related to that stock.

##Preprocessing_data.py:
###this file is generate the data for the modal,
###the strategy for the operate this data can be outlined as follows:
####(the operate details can be find in processData.ipynb)
####1, We iterate all of the  ‘r_price_train.csv’  in the ‘raw_price_train’ file and find the data field, through
####the data value we can find the news in that day
####2, we calculates the sentiment data of the news using the nltk library
####3, put the each day`s sentiment result into stock data filed

##Prepare_data_for_modal.py:
###1,prepare_data(): Normalize the stock data
###2,series_to_supervised(): transform a time series dataset into a supervised learning dataset. 

###exploratory_visualisation.py
###show all the graphs in the sequence that I plot to understand the data and the conclusion I made on the basis of them at that time

