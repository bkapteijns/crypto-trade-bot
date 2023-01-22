import ccxt
import talib
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

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

y = np.zeros(846)
# # Regression
# for i in range(100, 990):
#     y[i] = np.mean(X[i:i+10, 0])
# Classification
for i in range(100, 946):
    if change(df[i, 0], np.mean(df[i+1:i+21, 0])) > threshold:
        y[i-100] = 1
    elif change(df[i, 0], np.mean(df[i+1:i+21, 0])) < 0-threshold:
        y[i-100] = -1

X = np.zeros([846, 100, 18])
for i in range(846):
    X[i] = df[i:i+100]

X = X.reshape([846, 1800])

kf = StratifiedKFold(n_splits=3, shuffle=True)
errors1 = np.zeros((10, 5))
errors2 = np.zeros((10, 5))
for i in range(10):
    for j in range(5):
        errs1 = np.zeros(3)
        errs2 = np.zeros(3)
        for f, (train_index, test_index) in enumerate(kf.split(X, y)):
            model1 = RandomForestClassifier(
                n_estimators=i*50 + 1, criterion="gini", max_depth=j*4 + 1)
            model2 = RandomForestClassifier(
                n_estimators=i*50 + 1, criterion="entropy", max_depth=j*4 + 1)

            model1.fit(X[train_index], y[train_index])
            model2.fit(X[train_index], y[train_index])

            errs1[f] = np.mean(y[test_index] == model1.predict(X[test_index]))
            errs2[f] = np.mean(y[test_index] == model2.predict(X[test_index]))
        errors1[i][j] = np.mean(errs1)
        errors2[i][j] = np.mean(errs2)

f, ax = plt.subplots(1, 2)
ax[0].imshow(errors1)
ax[0].set_xlabel("n_estimators / 50")
ax[0].set_xlabel("max_depth / 4")
ax[1].imshow(errors2)
ax[1].set_xlabel("n_estimators / 50")
ax[1].set_xlabel("max_depth / 4")
plt.show()

# Random forest with default settings:
# - Gini coefficent splitting criterion (performance is similar, gini is a little faster)
# - max_depth = 10
# - n_estimators = 100

errs = np.zeros(8)
Z = np.arange(len(X))
np.random.shuffle(Z)
n_per_split = len(X)/8
for i in range(1, 9):
    newX = np.zeros((int(n_per_split*i), X.shape[1]))
    newy = np.zeros(int(n_per_split*i))
    for x in range((int(n_per_split*i))):
        newX[x] = X[Z[x]]
        newy[x] = y[Z[x]]

    for f, (train_index, test_index) in enumerate(kf.split(newX, newy)):
        model = RandomForestClassifier(
            n_estimators=100, criterion="gini", max_depth=10)

        model.fit(newX[train_index], newy[train_index])

        errs[i-1] = np.mean(newy[test_index] ==
                            model.predict(newX[test_index]))

plt.plot([105, 221, 317, 423, 528, 634, 740, 846], errs)
plt.show()

# More data still has a big increase in accuracy
