import os
import datetime
import urllib3
from dateutil.parser import parse
import threading
import pandas as pd

# assert 'QUANDL_KEY' in os.environ
import defines

import tushare as ts

from defines import *

ts.set_token(token_TS)
pro = ts.pro_api()
csi = pro.index_basic(market=id_market)


class stock_reader():
    def __init__(self):
        print("stockReader")

    def build_url(self, symbol):
        url = symbol
        return url


def download(i, symbol, url, output):
    df1 = ts.pro_bar(ts_code=url, adj='qfq', start_date="19890101", end_date="20020101")
    df2 = ts.pro_bar(ts_code=url, adj='qfq', start_date="20020101", end_date="20211111")

    # df1 = pro.index_daily(ts_code=url, start_date='19890101', end_date='20020101')   #取指数接口，没权限
    # df2 = pro.index_daily(ts_code=url, start_date='20020101', end_date='20211111')

    df = df2

    if df1 is not None:
        df = df1.append(df2)

    df.sort_values(by=FIELD_DATE, ascending=False, inplace=True)  # inplace is important

    df = df.reset_index(drop=True)

    print(df)
    fullPath = os.path.join(output, symbol)
    df.to_csv('{}.csv'.format(fullPath))
    print('download')


def download_all():
    reader = stock_reader()

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    for i, symbol in enumerate(stocks_list):
        url = reader.build_url(symbol)
        download(i, symbol, url, data_folder)


def download_stocks_list():
    data = pro.query('stock_basic',
                     exchange='',
                     list_status='L',
                     fields='ts_code,symbol,name,area,market,industry,list_date')
    print(data)
    data.to_csv('stocks_list.csv')


def download_daily(startIndex, endIndex):
    reader = stock_reader()

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    sl = pd.read_csv('stocks_list.csv',
                     header=0)
    for i in range(startIndex, endIndex):
        df = sl.iloc[i]
        #print(df.ts_code)
        symbol = df.ts_code.lower()
        print(symbol)
        url = reader.build_url(symbol)
        download(i, symbol, url, data_folder)


if __name__ == '__main__':
    download_daily(0, 5)
    # download_stocks_list()
    # download_all()
