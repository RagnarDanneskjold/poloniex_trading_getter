# poloniex_trading_getter
Collects trading information from the Cryptocurrency Trading Site Poloniex

rate.zip Includes an example of what can be constructed using this data gatherer. 
These ~8,750 scatterplots were created by looking at ~15 minute increments 
over the past year and and plotting the current rates of cryptocurrencies against each other.

After a quick test, I was unable to find a correlation between these scatterplots
and predicting an upwards or downwards trend in the following 10 minutes. I tested
this by running these scatterplots through a TensorFlow-built convolutional neural network.

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
    import matplotlib.image as mpimg
    import math
    from sklearn.preprocessing import scale
    from sklearn.preprocessing import StandardScaler
    from time import perf_counter as timer
    from sklearn.utils import shuffle
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn import svm
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import classification_report
