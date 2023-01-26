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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model1a = MLPClassifier(hidden_layer_sizes=(32,))
model1a.fit(X_test, y_test)
print("MLPC:", np.mean(y_test == model1a.predict(X_test)))

model1b = MLPClassifier(hidden_layer_sizes=(512, 128))
model1b.fit(X_train, y_train)
print("MLPC:", np.mean(y_test == model1b.predict(X_test)))

model2 = DecisionTreeClassifier(max_depth=10)
model2.fit(X_train, y_train)
print("DTC:", np.mean(y_test == model2.predict(X_test)))

model3 = RandomForestClassifier(max_depth=10)
model3.fit(X_train, y_train)
print("RFC:", np.mean(y_test == model3.predict(X_test)))

model4a = KNeighborsClassifier(n_neighbors=5)
model4a.fit(X_train, y_train)
print("KNN:", np.mean(y_test == model4a.predict(X_test)))

model4b = KNeighborsClassifier(n_neighbors=20)
model4b.fit(X_train, y_train)
print("KNN:", np.mean(y_test == model4b.predict(X_test)))

model4c = KNeighborsClassifier(n_neighbors=2)
model4c.fit(X_train, y_train)
print("KNN:", np.mean(y_test == model4c.predict(X_test)))

# Random forest classifier has been found to be the best performing model
