#数据预处理
#一共有8支股票，最后生成8个DataFrame格式的csv文件，把新闻和股票数据合在一起

import re
import os
import json
import datetime
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer


#把时间格式2014/01/01 替换成2014-01-01
def dateTool(dataString):
    strs = re.findall(r'\d+', dataString)
    a = datetime.date(int(strs[0]), int(strs[1]), int(strs[2]))
    return a.__format__('%Y-%m-%d')


# 读取一个文件里面新闻数据数据,返回text里面的数组，并且把数组里面的数据合成一个句子,并且过滤特殊字符
# 并且返回他的情感分析结果
def getNewsSentiment(path):
    try:
        with open(path, 'r', encoding='utf8') as fd:
            sentencesArr = [];
            lines = fd.readlines()
            for line in lines:  # 遍历数据
                data_list = json.loads(line)
                sentencesArr = sentencesArr + data_list['text']
            sentences = ' '.join(sentencesArr)
            # 正则表达式过滤特殊符号用空格符占位，双引号、单引号、句点、逗号
            pat_letter = re.compile(r'[^a-zA-Z \']+')
            sentences = pat_letter.sub('', sentences).strip().lower()
            # 通过NTLK分析句子情感
            sid = SentimentIntensityAnalyzer()
            ss = sid.polarity_scores(sentences)
            return ss
    except Exception as e:
        print('except Error: can not find the file->'+path)

#批量读取新闻和股票文件并且合并新闻和股票信息成一列
def processFinalData():
    BASE_URL = './stock_datasets/'
#     从股票数据获取到时间，然后作为新闻数据的文件夹目录，然后再去获取新闻的情感结果，然后合并成新的文件生成出来
#     stockPreFilePath = BASE_URL + 'preprocess_price_train/'
    stockFilePath = BASE_URL + 'raw_price_train/'
    newsFileBasePath = BASE_URL + 'tweet_train/'
    finalDataPath = BASE_URL + 'final_data/'
    files = os.listdir(stockFilePath)
    try:
        for file in files:
            priceFile = stockFilePath + file
            csv_data = pd.read_csv(priceFile)
            #数据只取2014-01-01之后的数据
            csv_data1 = csv_data.drop(range(333), axis=0, inplace=False)
            csv_data1["compound"] = '0'
            csv_data1["neg"] = '0'
            csv_data1["neu"] = '0'
            csv_data1["pos"] = '0'
            # 取得file的第一个数据，获得是第几支股票
            stockNum = file[0:1]
            newfileName = finalDataPath + 'pre_stock' + stockNum + '_data.csv'
            newsFilePath = newsFileBasePath + stockNum + '_tweet_train/'
            for data, row in csv_data1.T.iteritems():
                row_date = csv_data1.loc[data, 'Date']
                # 把时间格式2014/01/01 替换成2014-01-01
                format_date = dateTool(row_date)
                # 某支股票的新闻文件夹地址
                newsFilePath1 = newsFilePath + format_date
                # 取得csv data列根据时间取遍历新闻文件，然后csv文件后面追加Sentiment分析结果数据
                ss = getNewsSentiment(newsFilePath1)
                # extra columns to store the sentiment score
                if ss:
                    csv_data1.set_value(data, 'compound', ss['compound'])
                    csv_data1.set_value(data, 'neg', ss['neg'])
                    csv_data1.set_value(data, 'neu', ss['neu'])
                    csv_data1.set_value(data, 'pos', ss['pos'])
                csv_data1.to_csv(newfileName, index=False)

    except Exception as e:
         print('processFinalData ==', e)


processFinalData()

