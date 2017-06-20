import time
import sys
from poloniex import Poloniex
import numpy as np
from dateutil import parser
from coinbase.wallet.client import Client
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import normalize
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import librosa
import librosa.display
import matplotlib.image as mpimg
import math
from skimage import color
from skimage import io
from sklearn.preprocessing import scale

API_key = {key}
secret = {secret}

polo = Poloniex(API_key, secret)

def create_trade_data():
    trade_features = []
    trade_labels = []

    iterator = 0

    utc_time_now = time.mktime(datetime.datetime.utcnow().timetuple())
    label_time = int(time.time())
    sample_start = label_time - 2700 #look at 30 minute durations
    sample_end = label_time - 900 #look starting 15 minutes ago
    
    for i in range(70000): #70k samples
    #for i in range(30):

        #collect data
        one_data = [] #the trade data for one chunk
        
        try:
            trade_current = polo.marketTradeHist(currencyPair = 'USDT_XMR', start = label_time - 20000, end = label_time)[0]
            btc_current = polo.marketTradeHist(currencyPair = 'USDT_BTC', start = label_time - 20000, end = label_time)[0]
            
            trade_history = polo.marketTradeHist(currencyPair = 'USDT_XMR', start = sample_start, end = sample_end)
            usdt_btc_trade_history = polo.marketTradeHist(currencyPair = 'USDT_BTC', start = sample_start, end = sample_end)
            btc_xmr_trade_history = polo.marketTradeHist(currencyPair = 'BTC_XMR', start = sample_start, end = sample_end)
            xmr_dash_trade_history = polo.marketTradeHist(currencyPair = 'XMR_DASH', start = sample_start, end = sample_end)
            xmr_ltc_trade_history = polo.marketTradeHist(currencyPair = 'XMR_LTC', start = sample_start, end = sample_end)

            rate = []
            btc_rate = []

            btc_xmr_buys = 0
            btc_xmr_bought = 0
            btc_xmr_sells = 0
            btc_xmr_sold = 0

            xmr_dash_buys = 0
            xmr_dash_bought = 0
            xmr_dash_sells = 0
            xmr_dash_sold = 0

            xmr_ltc_buys = 0
            xmr_ltc_bought = 0
            xmr_ltc_sells = 0
            xmr_ltc_sold = 0

            if len(trade_history) <= 4000 and len(trade_history) >= 2:
                for trade in range(len(trade_history)): #look at all trades made in a 30 minute window
                    ###RATE###
                    moment_rate = float(trade_history[trade]['rate'])
                    rate.append(moment_rate)

                    btc_moment_rate = float(usdt_btc_trade_history[trade]['rate'])
                    btc_rate.append(btc_moment_rate)

                ###BUYS AND SELLS###            
                #NUMBER OF BUYS AND TOTAL AMOUNT BOUGHT 
                for trade in range(len(btc_xmr_trade_history)):
                    temp_btc_xmr_buys, temp_btc_xmr_bought, temp_btc_xmr_sells, temp_btc_xmr_sold = bought_sold(btc_xmr_trade_history, trade)
                    btc_xmr_buys += temp_btc_xmr_buys
                    btc_xmr_bought += temp_btc_xmr_bought
                    btc_xmr_sells += temp_btc_xmr_sells
                    btc_xmr_sold += temp_btc_xmr_sold

                for trade in range(len(xmr_dash_trade_history)):
                    temp_xmr_dash_buys, temp_xmr_dash_bought, temp_xmr_dash_sells, temp_xmr_dash_sold = bought_sold(xmr_dash_trade_history, trade)
                    xmr_dash_buys += temp_xmr_dash_buys
                    xmr_dash_bought += temp_xmr_dash_bought
                    xmr_dash_sells += temp_xmr_dash_sells
                    xmr_dash_sold += temp_xmr_dash_sold

                for trade in range(len(xmr_ltc_trade_history)):
                    temp_xmr_ltc_buys, temp_xmr_ltc_bought, temp_xmr_ltc_sells, temp_xmr_ltc_sold = bought_sold(xmr_ltc_trade_history, trade)
                    xmr_ltc_buys += temp_xmr_ltc_buys
                    xmr_ltc_bought += temp_xmr_ltc_bought
                    xmr_ltc_sells += temp_xmr_ltc_sells
                    xmr_ltc_sold += temp_xmr_ltc_sold

                one_data.append(btc_xmr_buys)
                one_data.append(btc_xmr_bought)
                one_data.append(btc_xmr_sells)
                one_data.append(btc_xmr_sold)

                one_data.append(xmr_dash_buys)
                one_data.append(xmr_dash_bought)
                one_data.append(xmr_dash_sells)
                one_data.append(xmr_dash_sold)

                one_data.append(xmr_ltc_buys)
                one_data.append(xmr_ltc_bought)
                one_data.append(xmr_ltc_sells)
                one_data.append(xmr_ltc_sold)

                #Rate
                #using a list of all the rates, calculate the rate change, mean, and standard deviation
                oldest_rate = rate[-1]
                newest_rate = rate[0]
                rate_percent_change = (newest_rate - oldest_rate) / oldest_rate
                one_data.append(rate_percent_change)

                rate = np.array(rate)
                rate_mean = np.mean(rate)
                rate_std = np.std(rate)
                one_data.append(rate_mean)
                one_data.append(rate_std)

                oldest_btc_rate = btc_rate[-1]
                newest_btc_rate = btc_rate[0]
                rate_btc_percent_change = (newest_btc_rate - oldest_btc_rate) / oldest_btc_rate
                one_data.append(rate_btc_percent_change)

                btc_rate = np.array(btc_rate)
                rate_btc_mean = np.mean(btc_rate)
                rate_btc_std = np.std(btc_rate)
                one_data.append(rate_btc_mean)
                one_data.append(rate_btc_std)

                #Volume
                #Using the total amount sold and bought, and the time between the first and last purchase, calculate volume
                vol2 = volume(btc_xmr_trade_history, btc_xmr_bought, btc_xmr_sold)
                vol3 = volume(xmr_dash_trade_history, xmr_dash_bought, xmr_dash_sold)
                vol4 = volume(xmr_ltc_trade_history, xmr_ltc_bought, xmr_ltc_sold)

                one_data.append(vol2)
                one_data.append(vol3)
                one_data.append(vol4)

                trade_features.append(one_data)
                ###LABELS###
                #add the label to trade_labels
                #find the current price

                rate_new = float(trade_current['rate'])
                rate_old = float(trade_history[0]['rate'])

                percent_change = (rate_new - rate_old) / rate_old
                trade_labels.append(percent_change)

            label_time = label_time - 660 #go back 11 minutes
            sample_start = label_time - 2700
            sample_end = label_time - 900

            iterator += 1
            percent = int(iterator / 700)

            sys.stdout.write("\r" + str(percent) + " percent complete" + " ||| " + str(trade_current['date']))
            sys.stdout.flush()

        except:
            label_time = label_time - 660 #go back 11 minutes
            sample_start = label_time - 2700
            sample_end = label_time - 900

            iterator += 1
            percent = int(iterator / 700)

            sys.stdout.write("\r" + str(percent) + " percent complete" + " ||| " + str(trade_current['date']))
            sys.stdout.flush()

    print("")
    print("Finished")
    print("")
    trade_features = np.array(trade_features)
    trade_labels = np.array(trade_labels)
    return trade_features, trade_labels
            
trade_features, trade_labels = create_trade_data()

print(trade_features.shape)
print(trade_labels.shape)

feature_names = ['btc_xmr_buys', 'btc_xmr_bought', 'btc_xmr_sells', 'btc_xmr_sold', 'xmr_dash_buys', 'xmr_dash_bought',
                 'xmr_dash_sells', 'xmr_dash_sold', 'xmr_ltc_buys', 'xmr_ltc_bought', 'xmr_ltc_sells', 'xmr_ltc_sold',
                 'rate_percent_change', 'rate_mean', 'rate_std', 'btc_rate_percent_change', 'btc_rate_mean', 'btc_rate_std',
                 'btc_xmr_vol', 'xmr_dash_vol', 'xmr_ltc_vol']

label_names = ['percent_change']

trade_features_df = pd.DataFrame(trade_features, columns = feature_names)
trade_labels_df = pd.DataFrame(trade_labels, columns = label_names)
