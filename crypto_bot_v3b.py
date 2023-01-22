import ccxt
import talib
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_tabular
import shap
import eli5

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

n_indicators = 10
aggression = 4  # Aggression variable from 1 to n_indicators/2
threshold = 0.08   # What percentage change is valid

print("Balance:\t", "BTC:", exchange.fetch_balance()[
      "BTC"], "\n\t\t", "USD:", exchange.fetch_balance()["USDT"])

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


def change(old, new):
    return ((new - old)/old) * 100


# Training the model
source = "\\data_gathering\\store.csv"
ohlcv = pd.read_csv(os.getcwd() + source, header=None).to_numpy()
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
df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
df['mfi'] = talib.MFI(df['high'], df['low'],
                      df['close'], df['volume'], timeperiod=14)
df['obv'] = talib.OBV(df['close'], df['volume'])
df['sar'] = talib.SAR(df['high'], df['low'],
                      acceleration=0.02, maximum=0.2)

df = df.to_numpy()
# first 4 columns are timestamp, open, high, and low; some indicators use previous data to compute (so you get NANs)
df = df[33:, 4:]

y = np.zeros(len(df) - 100)
for i in range(100, len(df)):
    if change(df[i, 0], np.mean(df[i+1:i+21, 0])) > threshold:
        y[i-100] = 1
    elif change(df[i, 0], np.mean(df[i+1:i+21, 0])) < 0-threshold:
        y[i-100] = -1

X = np.zeros([len(df) - 100, 100, 18])
for i in range(len(df) - 100):
    X[i] = df[i:i+100]

X = X.reshape([len(df) - 100, 18 * 100])

model = RandomForestClassifier(
    n_estimators=100, criterion="gini", max_depth=10)
model.fit(X, y)


while(True):  # Fetch historical data
    ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=133)
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
    df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
    df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['mfi'] = talib.MFI(df['high'], df['low'],
                          df['close'], df['volume'], timeperiod=14)
    df['obv'] = talib.OBV(df['close'], df['volume'])
    df['sar'] = talib.SAR(df['high'], df['low'],
                          acceleration=0.02, maximum=0.2)

    X = df.to_numpy()
    X = np.array([X[33:, 4:].reshape(1800,)])

    if model.predict(X) == 1:
        # Buy signal
        if(not long):
            long = True
            short = False
            print("Bought at:", exchange.fetch_ticker(symbol)["ask"])
            while (bitcoins < 1):
                # exchange.create_market_buy_order(symbol, 1)
                buy(exchange.fetch_ticker(symbol)["ask"])
    elif model.predict(X) == -1:
        # Sell signal
        if (not short):
            long = False
            short = True
            print("Sold at:", exchange.fetch_ticker(symbol)["bid"])
            while (bitcoins > -1):
                # exchange.create_market_sell_order(symbol, 1)
                sell(exchange.fetch_ticker(symbol)["bid"])
    else:
        # Do nothing
        print("Hold")
        if (long):
            print("Sold at:", exchange.fetch_ticker(symbol)["bid"])
            # exchange.create_market_sell_order(symbol, 1)
            sell(exchange.fetch_ticker(symbol)["bid"])
        if (short):
            print("Bought at:", exchange.fetch_ticker(symbol)["ask"])
            # exchange.create_market_buy_order(symbol, 1)
            buy(exchange.fetch_ticker(symbol)["ask"])
        long = False
        short = False
        pass
    time.sleep(59.9)
    print("Value:", capital + bitcoins * exchange.fetch_ticker(symbol)["last"])
