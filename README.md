# poloniex_trading_getter
Collects trading information from the Cryptocurrency Trading Site Poloniex

rate.zip Includes an example of what can be constructed using this data gatherer. 
These ~8,750 scatterplots were created by looking at ~15 minute increments 
over the past year and and plotting the current rates of cryptocurrencies against each other.

After a quick test, I was unable to find a correlation between these scatterplots
and predicting an upwards or downwards trend in the following 10 minutes. I tested
this by running these scatterplots through a TensorFlow-built convolutional neural network.
