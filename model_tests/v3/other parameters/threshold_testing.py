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
threshold = 0.05   # What percentage change is valid

print("Balance:\t", "BTC:", exchange.fetch_balance()[
      "BTC"], "\n\t\t", "USD:", exchange.fetch_balance()["USDT"])

long = False
short = False

capital = np.array([30000]*10)
bitcoins = np.zeros(10)


def buy(i, price):
    global capital, bitcoins
    capital[i] -= price
    bitcoins[i] += 1


def sell(i, price):
    global capital, bitcoins
    capital[i] += price
    bitcoins[i] -= 1


def change(old, new):
    return ((new - old)/old) * 100


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

profits = np.zeros((5, (int(len(df)/3)+1), 10))
for threshold in np.arange(0.02, 0.22, 0.02):
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

    kf = KFold(n_splits=3)
    for f, (train_indices, test_indices) in enumerate(kf.split(X, y)):
        model = RandomForestClassifier(
            n_estimators=100, criterion="gini", max_depth=10)
        model.fit(X[train_indices], y[train_indices])
        for i in range(len(test_indices)):
            if (model.predict(X[[test_indices[i]]]) == 1 and bitcoins[int(threshold*25-1)] != 1):
                buy(int(threshold*25-1), X[test_indices[i]][0])
            elif (model.predict(X[[test_indices[i]]]) == -1 and bitcoins[int(threshold*25-1)] != -1):
                sell(int(threshold*25-1), X[test_indices[i]][0])
            elif (bitcoins[int(threshold*25-1)] == 1):
                sell(int(threshold*25-1), X[test_indices[i]][0])
            elif (bitcoins[int(threshold*25-1)] == -1):
                buy(int(threshold*25-1), X[test_indices[i]][0])
            profits[f, i, int(threshold*25-1)] = capital[int(threshold*25-1)] + \
                bitcoins[int(threshold*25-1)]*X[test_indices[i]][0]
        capital = np.array([30000]*10)
        bitcoins = np.zeros(10)

profits = np.mean(profits, axis=0)

for threshold in np.arange(0.04, 0.44, 0.04):
    plt.plot(range(int(len(df)/3)+1),
             profits[:, int(threshold*25-1)], label=str(threshold))
plt.legend()
plt.show()

# 0.08 threshold gives the best yield
