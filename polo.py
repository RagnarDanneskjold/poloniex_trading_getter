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

    iterator = 0

    utc_time_now = time.mktime(datetime.datetime.utcnow().timetuple())
    label_time = int(time.time())
    sample_start = label_time - 1200 #look at 15 minute durations
    sample_end = label_time - 300 #look starting 5 minutes ago
    
    for i in range(30000): #30k samples
    #for i in range(30):

        #collect data
        one_data = [] #the trade data for one chunk
        try:
            trade_history = polo.marketTradeHist(start = sample_start, end = sample_end)

            rate = []
            trade_times = []
            buys = 0
            sells = 0
            amt_bought = 0
            amt_sold = 0
            if len(trade_history) <= 4000 and len(trade_history) >= 2:
                for trade in range(len(trade_history)): #look at all trades made in a 15 minute window
                    #Do the following: collect all the rates (list), all the times (list), sum of buys, 
                    #sum of sells, # of buys, # of sells

                    ###RATE###
                    moment_rate = float(trade_history[trade]['rate'])
                    rate.append(moment_rate)

                    ###TIME OF DAY###
                    #find average trade time
                    tradetime = parser.parse(trade_history[trade]['date'])
                    tradetime_hours = tradetime.hour * 60 * 60
                    tradetime_mins = tradetime.minute * 60
                    tradetime_secs = tradetime.second
                    total_secs = tradetime_secs + tradetime_mins + tradetime_hours
                    trade_times.append(total_secs)           

                    ###BUYS AND SELLS###            
                    #NUMBER OF BUYS AND TOTAL AMOUNT BOUGHT
                    if trade_history[trade]['type'] == 'buy':
                        buys += 1
                        amt_bought += float(trade_history[trade]['amount'])
                    #NUMBER OF SELLS AND TOTAL AMOUNT SOLD
                    else:
                        sells += 1
                        amt_sold += float(trade_history[trade]['amount'])

                one_data.append(buys)
                one_data.append(sells)
                one_data.append(amt_bought)
                one_data.append(amt_sold)

                #Rate
                #using a list of all the rates, calculate the min, max, and mean

                min_rate = min(rate)
                max_rate = max(rate)
                mean_rate = float(sum(rate)) / float(len(rate))
                one_data.append(min_rate)
                one_data.append(max_rate)
                one_data.append(mean_rate)

                #Volume
                #Using the total amount sold and bought, and the time between the first and last purchase, calculate volume
                total_amt = amt_bought + amt_sold
                trade_old = parser.parse(trade_history[len(trade_history) - 1]['date'])
                trade_new = parser.parse(trade_history[0]['date'])
                trade_diff = (trade_new - trade_old).total_seconds()
                try:
                    volume = total_amt / float(trade_diff)
                except:
                    volume = 0
                one_data.append(volume)

                #Time of day
                #Using a list of the time of every transaction, find the average and classify it as being in one of 8 categories

                mean_time = float(sum(trade_times)) / float(len(trade_times))
                if mean_time >= 0 and mean_time < 10800:
                    time_of_day = 0
                elif mean_time >= 10800 and mean_time < (10800 * 2):
                    time_of_day = 1
                elif mean_time >= (10800 * 2) and mean_time < (10800 * 3):
                    time_of_day = 2
                elif mean_time >= (10800 * 3) and mean_time < (10800 * 4):
                    time_of_day = 3
                elif mean_time >= (10800 * 4) and mean_time < (10800 * 5):
                    time_of_day = 4
                elif mean_time >= (10800 * 5) and mean_time < (10800 * 6):
                    time_of_day = 5
                elif mean_time >= (10800 * 6) and mean_time < (10800 * 7):
                    time_of_day = 6
                else:
                    time_of_day = 7
                one_data.append(time_of_day)

                trade_features.append(one_data)

            label_time = label_time - 900 #go back 15 minutes
            sample_start = label_time - 1200
            sample_end = label_time - 300

            iterator += 1
            percent = int(iterator / 300)

            sys.stdout.write("\r" + str(percent) + " percent complete" + " ||| " + str(trade_current['date']))
            sys.stdout.flush()

        except: #in case of running into an error communicating with the server
            label_time = label_time - 900 #go back 15 minutes
            sample_start = label_time - 1200
            sample_end = label_time - 300

            iterator += 1
            percent = int(iterator / 300)

            sys.stdout.write("\r" + str(percent) + " percent complete" + " ||| " + str(trade_current['date']))
            sys.stdout.flush()
            
    print("")
    print("Finished")
    print("")
    trade_features = np.array(trade_features)
    return trade_features
