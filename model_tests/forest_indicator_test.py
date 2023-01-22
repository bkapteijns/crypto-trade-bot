import ccxt
import talib
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier

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
threshold = 0.05   # What percentage change is valid

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


source = "\\data_gathering\\store.csv"
ohlcv = pd.read_csv(os.getcwd() + source, header=None).to_numpy()
df = pd.DataFrame(
    ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
print(ohlcv.shape)
print(df.shape)

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

y = np.zeros(len(df) - 133)
# # Regression
# for i in range(100, 990):
#     y[i] = np.mean(X[i:i+10, 0])
# Classification
for i in range(100, len(df) - 33):
    if change(df[i, 0], np.mean(df[i+1:i+21, 0])) > threshold:
        y[i-100] = 1
    elif change(df[i, 0], np.mean(df[i+1:i+21, 0])) < 0-threshold:
        y[i-100] = -1

best_X = np.zeros([len(df) - 133, 100, 12])
okay_X = np.zeros([len(df) - 133, 100, 16])
all_X = np.zeros([len(df) - 133, 100, 18])
for i in range(len(df) - 133):
    indices = np.arange(18)
    best_X[i] = df[i:i+100, np.delete(indices, [11, 12, 14, 15, 16, 17])]
    okay_X[i] = df[i:i+100, np.delete(indices, [14, 17])]
    all_X[i] = df[i:i+100]

best_X = best_X.reshape([len(df) - 133, 1200])
okay_X = okay_X.reshape([len(df) - 133, 1600])
all_X = all_X.reshape([len(df) - 133, 1800])

print("y", y.shape)
print("best_X", best_X.shape)
print("okay_X", okay_X.shape)
print("all_X", all_X.shape)

kf = StratifiedKFold(n_splits=3, shuffle=True)
best_errs = np.zeros(3)
okay_errs = np.zeros(3)
all_errs = np.zeros(3)

for f, (train_index, test_index) in enumerate(kf.split(all_X, y)):
    best_model = RandomForestClassifier(
        n_estimators=100, criterion="gini", max_depth=10)
    okay_model = RandomForestClassifier(
        n_estimators=100, criterion="entropy", max_depth=10)
    all_model = RandomForestClassifier(
        n_estimators=100, criterion="entropy", max_depth=10)

    best_model.fit(best_X[train_index], y[train_index])
    okay_model.fit(okay_X[train_index], y[train_index])
    all_model.fit(all_X[train_index], y[train_index])

    best_errs[f] = np.mean(
        y[test_index] == best_model.predict(best_X[test_index]))
    okay_errs[f] = np.mean(
        y[test_index] == okay_model.predict(okay_X[test_index]))
    all_errs[f] = np.mean(
        y[test_index] == all_model.predict(all_X[test_index]))

print(best_errs)
print(okay_errs)
print(all_errs)


best_errs = np.zeros(3)
okay_errs = np.zeros(3)
all_errs = np.zeros(3)

# All datasets perform more or less equally, so it doesn't matter if we pick only the best indicators or all indicators
