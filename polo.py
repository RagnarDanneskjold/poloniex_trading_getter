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
