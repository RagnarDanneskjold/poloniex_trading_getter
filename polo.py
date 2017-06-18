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
    
    for i in range(40000): #30k samples
    #for i in range(30):

        #collect data
        one_data = [] #the trade data for one chunk
        #trade_current = polo.marketTradeHist(currencyPair = 'BTC_XMR', start = label_time - 20000, end = label_time)[0]
        trade_current = polo.marketTradeHist(currencyPair = 'USDT_XMR', start = label_time - 20000, end = label_time)[0]
        
        try:
            #look at the data from 10 minutes ago to 20 minutes ago
            #trade_history = polo.marketTradeHist(currencyPair = 'BTC_XMR', start = sample_start, end = sample_end)
            trade_history = polo.marketTradeHist(currencyPair = 'USDT_XMR', start = sample_start, end = sample_end)
            btc_xmr_trade_history = polo.marketTradeHist(currencyPair = 'BTC_XMR', start = sample_start, end = sample_end)
            eth_xmr_trade_history = polo.marketTradeHist(currencyPair = 'XMR_DASH', start = sample_start, end = sample_end)
            btc_eth_trade_history = polo.marketTradeHist(currencyPair = 'XMR_LTC', start = sample_start, end = sample_end)



            rate = []
            trade_times = []

            btc_xmr_buys = 0
            btc_xmr_bought = 0
            btc_xmr_sells = 0
            btc_xmr_sold = 0

            eth_xmr_buys = 0
            eth_xmr_bought = 0
            eth_xmr_sells = 0
            eth_xmr_sold = 0

            btc_eth_buys = 0
            btc_eth_bought = 0
            btc_eth_sells = 0
            btc_eth_sold = 0

            if len(trade_history) <= 4000 and len(trade_history) >= 2:
                for trade in range(len(trade_history)): #look at all trades made in a 15 minute window
                    ###RATE###
                    moment_rate = float(trade_history[trade]['rate'])
                    rate.append(moment_rate)

                ###BUYS AND SELLS###            
                #NUMBER OF BUYS AND TOTAL AMOUNT BOUGHT 
                for trade in range(len(btc_xmr_trade_history)):
                    temp_btc_xmr_buys, temp_btc_xmr_bought, temp_btc_xmr_sells, temp_btc_xmr_sold = bought_sold(btc_xmr_trade_history, trade)
                    btc_xmr_buys += temp_btc_xmr_buys
                    btc_xmr_bought += temp_btc_xmr_bought
                    btc_xmr_sells += temp_btc_xmr_sells
                    btc_xmr_sold += temp_btc_xmr_sold

                for trade in range(len(eth_xmr_trade_history)):
                    temp_eth_xmr_buys, temp_eth_xmr_bought, temp_eth_xmr_sells, temp_eth_xmr_sold = bought_sold(eth_xmr_trade_history, trade)
                    eth_xmr_buys += temp_eth_xmr_buys
                    eth_xmr_bought += temp_eth_xmr_bought
                    eth_xmr_sells += temp_eth_xmr_sells
                    eth_xmr_sold += temp_eth_xmr_sold

                for trade in range(len(btc_eth_trade_history)):
                    temp_btc_eth_buys, temp_btc_eth_bought, temp_btc_eth_sells, temp_btc_eth_sold = bought_sold(btc_eth_trade_history, trade)
                    btc_eth_buys += temp_btc_eth_buys
                    btc_eth_bought += temp_btc_eth_bought
                    btc_eth_sells += temp_btc_eth_sells
                    btc_eth_sold += temp_btc_eth_sold

                one_data.append(btc_xmr_buys)
                one_data.append(btc_xmr_bought)
                one_data.append(btc_xmr_sells)
                one_data.append(btc_xmr_sold)

                one_data.append(eth_xmr_buys)
                one_data.append(eth_xmr_bought)
                one_data.append(eth_xmr_sells)
                one_data.append(eth_xmr_sold)

                one_data.append(btc_eth_buys)
                one_data.append(btc_eth_bought)
                one_data.append(btc_eth_sells)
                one_data.append(btc_eth_sold)

                #Rate
                #using a list of all the rates, calculate the min, max, and mean
                min_rate = min(rate)
                max_rate = max(rate)
                rate_difference = max_rate - min_rate
                one_data.append(rate_difference)

                #Volume
                #Using the total amount sold and bought, and the time between the first and last purchase, calculate volume
                vol2 = volume(btc_xmr_trade_history, btc_xmr_bought, btc_xmr_sold)
                vol3 = volume(eth_xmr_trade_history, eth_xmr_bought, eth_xmr_sold)
                vol4 = volume(btc_eth_trade_history, btc_eth_bought, btc_eth_sold)

                one_data.append(vol2)
                one_data.append(vol3)
                one_data.append(vol4)

                trade_features.append(one_data)
                ###LABELS###
                #add the label to trade_labels
                #find the current price

                rate_new = float(trade_current['rate'])
                rate_old = float(trade_history[0]['rate'])

                rate_increase = (rate_new - rate_old) / rate_old
                rate_decrease = (rate_old - rate_new) / rate_old

                if rate_increase >= 0.01: #if the price is projected to go up by at least 1 percent
                    trade_labels.append([1, 0, 0])

                elif rate_decrease >= 0.01: #if the price is projected to go down by at least 1 percent
                    trade_labels.append([0, 0, 1])

                else: #if the price does not go above or below one percent
                    trade_labels.append([0, 1, 0])

            label_time = label_time - 660 #go back 11 minutes
            sample_start = label_time - 2700
            sample_end = label_time - 900

            iterator += 1
            percent = int(iterator / 400)

            sys.stdout.write("\r" + str(percent) + " percent complete" + " ||| " + str(trade_current['date']))
            sys.stdout.flush()

        except:
            label_time = label_time - 660 #go back 11 minutes
            sample_start = label_time - 2700
            sample_end = label_time - 900

            iterator += 1
            percent = int(iterator / 400)

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
