import pandas as pd 
import time
import seaborn as sns
import matplotlib.pyplot as plt 
import datetime
import numpy as np

def calculate_close_off_high(row):
    """
    Normalized Data
    Algorithm: a + (((x - A)(b - a))/(B - A))
    a = -1, b = 1 => (b-a = 2)
    A = Daily High, B = Daily Low
    X = Close Price
    Close_Off_High = Close - High Price of Day
    """
    high, low, close = row['High'], row['Low'], row['Close']
    numerator = 2*(close - high)
    denominator = (low - high)
    return (-1) + (numerator / float(denominator))

def calculate_volatility(row):
    """
    Normalized Data 
    Algorithm: (a - b)/c
    a = high, b = low, c = open_price
    """
    high, low, open_price = row['High'], row['Low'], row['Open']
    numerator = high - low 
    return numerator/float(open_price)

def get_crypto_data(crypto_name):
    # get market info for bitcoin from the start of 2016 to the current day
    url = "https://coinmarketcap.com/currencies/{crypto}/historical-data/?start=20170428&end=".format(crypto=crypto_name)+time.strftime("%Y%m%d")
    market_info = pd.read_html(url)[0]
    # convert the date string to the correct date format
    market_info = market_info.assign(Date=pd.to_datetime(market_info['Date']))
    # convert high, low, open, close to numbers
    market_info['High'] = market_info['High'].astype('int64')
    market_info['Low'] = market_info['Low'].astype('int64')
    market_info['Open'] = market_info['Open'].astype('int64')
    market_info['Close'] = market_info['Close'].astype('int64')
    # create new series of dataframes
    model_info = market_info[['Date', 'Close', 'Volume']].copy()
    # add new columns
    model_info['Close_Off_High'] = market_info.apply(calculate_close_off_high, axis=1)
    model_info['Volatility'] = market_info.apply(calculate_volatility, axis=1)
    # when Volume is equal to '-' convert it to 0
    # Filter Series
    # filtered = market_info.filter(items=['Volume'])
    # filtered = map(lambda vol: vol_fixer(vol), filtered)
    # print filtered
    # print market_info.loc[(market_info['Volume'] == '-') , 'Volume']
    # convert to int
    model_info['Volume'] = model_info['Volume'].astype('int64')
    # look at the first few rows
    return model_info.head()

print get_crypto_data('bitcoin')