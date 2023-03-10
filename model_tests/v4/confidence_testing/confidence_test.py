# imports
import ccxt
import talib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

## Create the exchange
exchange = ccxt.binance()
symbol = "BTC/USDT"

## Set API keys
api_key = open("api_key", "r")
exchange.apiKey = api_key.read()
api_key.close()
secret = open("secret", "r")
exchange.secret = secret.read()
secret.close()

# Bot values
confidence = 0.2
threshold = 0
past = 100
future = 20
tree_depth = 20
n_trees = 100

# Trade values
long = False
short = False
capital = 30000
bitcoins = 0

# Buy and sell functions
def buy(price):
    global capital, bitcoins
    capital -= price
    bitcoins += 1


def sell(price):
    global capital, bitcoins
    capital += price
    bitcoins -= 1


def change(old, new):
    return ((new - old) / old) * 100


import sys

sys.path.append(os.getcwd())
from data_gathering.data_merger_v4 import merge_data

data = merge_data(["store1.csv", "store.csv"])

X = np.array(data["X"])
y = np.array(data["y"])

print(X.shape)
print(y.shape)

# Training the model
model = RandomForestClassifier(
    n_estimators=n_trees, criterion="gini", max_depth=tree_depth
)
model.fit(X, y)


#### Getting the test data
source = "\\data_gathering\\store2.csv"
ohlcv = pd.read_csv(os.getcwd() + source, header=None).to_numpy()
df = pd.DataFrame(
    ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
)

high_values = df["high"][133:].to_numpy()
low_values = df["low"][133:].to_numpy()

# Calculate indicators
df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
    df["close"], fastperiod=12, slowperiod=26, signalperiod=9
)
df["rsi"] = talib.RSI(df["close"], timeperiod=14)
df["bollinger_upper"], df["bollinger_middle"], df["bollinger_lower"] = talib.BBANDS(
    df["close"], timeperiod=20
)
df["stoch_k"], df["stoch_d"] = talib.STOCH(
    df["high"], df["low"], df["close"], fastk_period=14, slowk_period=3, slowd_period=3
)
df["ema_12"] = talib.EMA(df["close"], timeperiod=12)
df["ema_26"] = talib.EMA(df["close"], timeperiod=26)
df["adx"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
df["cci"] = talib.CCI(df["high"], df["low"], df["close"], timeperiod=14)
df["mfi"] = talib.MFI(df["high"], df["low"], df["close"], df["volume"], timeperiod=14)
df["obv"] = talib.OBV(df["close"], df["volume"])
df["sar"] = talib.SAR(df["high"], df["low"], acceleration=0.02, maximum=0.2)

# Formatting the data
df = df.to_numpy()
# first 4 columns are timestamp, open, high, and low; some indicators use previous data to compute (so you get NANs)
df = df[33:, 4:]

X = np.zeros([len(df) - past, past, 18])
for i in range(len(df) - past):
    X[i] = df[i : i + past]

X = X.reshape([len(df) - past, 18 * past])

## Perform the trading
vals = []

for confidence in [0.1, 0.2, 0.3, 0.4]:
    for close_confidence in [0, -0.1, -0.2, -0.3, -0.4, -0.5]:
        capital = 30000
        bitcoins = 0
        values = []
        for i in range(len(df) - past):
            if model.predict_proba([X[i]])[0][1] >= 0.5 + confidence:
                if not long:
                    long = True
                    short = False
                    while bitcoins < 1:
                        buy(df[i, 0] + 1)
            elif model.predict_proba([X[i]])[0][0] >= 0.5 + confidence:
                # Sell signal
                if not short:
                    long = False
                    short = True
                    while bitcoins > -1:
                        sell(df[i, 0] - 1)
            else:
                if long and model.predict_proba([X[i]])[0][1] >= 0.5 + close_confidence:
                    sell(df[i, 0] - 1)
                if (
                    short
                    and model.predict_proba([X[i]])[0][0] >= 0.5 + close_confidence
                ):
                    buy(df[i, 0] + 1)
                long = False
                short = False
            values.append(capital + bitcoins * df[i, 0])
        print(values)
        vals.append(values)

print(vals)

f, ax = plt.subplots(6, 4)
for i in range(4):
    for j in range(6):
        ax[j, i].plot(vals[i * 6 + j])
plt.show()
