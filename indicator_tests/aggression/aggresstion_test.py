import ccxt
import talib
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Initialize the bot
exchange = ccxt.binance()
symbol = 'BTC/USDT'

# Test phase
# exchange.set_sandbox_mode(True)
api_key = open("api_key", "r")
exchange.apiKey = api_key.read()
api_key.close()
secret = open("secret", "r")
exchange.secret = secret.read()
secret.close()
# print(exchange.urls)

n_indicators = 5
aggression = 5  # Aggression variable from 1 to n_indicators/2

long = False
short = False

capital = 30000
bitcoins = 0


def buy(price):
    global capital, bitcoins
    capital -= price
    bitcoins += 1


def sell(price):
    global capital, bitcoins
    capital += price
    bitcoins -= 1


ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=1000)
df = pd.DataFrame(
    ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Calculate indicators
df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
    df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['rsi'] = talib.RSI(df['close'], timeperiod=14)
df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = talib.BBANDS(
    df['close'], timeperiod=20)
df['stoch_k'], df['stoch_d'] = talib.STOCH(
    df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

profits = np.zeros(900)

# Define trading strategy
for i in range(100, 1000):  # Fetch historical data
    if np.sum([(df['macd'][i] > df['macd_signal'][i] + 7), (df['rsi'][i] > 50 + 2), (df['close'][i] > df['bollinger_middle'][i] * 1.0007), (df['stoch_k'][i] > df['stoch_d'][i] + 7), (df['adx'][i] > 25 + 7)]) > n_indicators - aggression:
        # Buy signal
        if(not long):
            long = True
            short = False
            while (bitcoins < 1):
                # exchange.create_market_buy_order(symbol, 1)
                buy(df["close"][i])
    elif np.sum([(df['macd'][i] < df['macd_signal'][i] - 7), (df['rsi'][i] < 50 - 2), (df['close'][i] < df['bollinger_middle'][i] * 0.9993), (df['stoch_k'][i] < df['stoch_d'][i] - 7), (df['adx'][i] < 25 - 7)]) > n_indicators - aggression:
        # Sell signal
        if (not short):
            long = False
            short = True
            while (bitcoins > -1):
                # exchange.create_market_sell_order(symbol, 1)
                sell(df["close"][i])
    else:
        # Do nothing
        if (long):
            # exchange.create_market_sell_order(symbol, 1)
            sell(df["close"][i])
        if (short):
            # exchange.create_market_buy_order(symbol, 1)
            buy(df["close"][i])
        long = False
        short = False
        pass
    profits[i-100] = capital + df["close"][i]*bitcoins

plt.plot(range(900), profits)
plt.show()
