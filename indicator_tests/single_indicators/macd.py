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

n_indicators = 10
aggression = 3  # Aggression variable from 1 to n_indicators/2

long = False
short = False

capital = np.array([30000]*5)
bitcoins = np.zeros(5)

ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=1000)
df = pd.DataFrame(
    ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Calculate indicators
df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
    df['close'], fastperiod=12, slowperiod=26, signalperiod=9)


def buy(method, i):
    global capital, bitcoins
    price = df["close"][i]
    capital[method] -= price
    bitcoins[method] += 1


def sell(method, i):
    global capital, bitcoins
    price = df["close"][i]
    capital[method] += price
    bitcoins[method] -= 1


profits = np.zeros((900, 5))

# Define trading strategy
for i in range(100, 1000):  # Fetch historical data
    x = 0

    if (df['macd'][i] > df['macd_signal'][i]):
        while (bitcoins[x] < 1):
            buy(x, i)
    else:
        while(bitcoins[x] > -1):
            sell(x, i)
    profits[i-100, x] = capital[x] + (bitcoins[x] * df["close"][i])
    x += 1

    if (df['macd'][i] > df['macd_signal'][i] + 2):
        while (bitcoins[x] < 1):
            buy(x, i)
    elif (df['macd'][i] < df['macd_signal'][i] - 2):
        while(bitcoins[x] > -1):
            sell(x, i)
    elif (bitcoins[x] == 1):
        sell(x, i)
    elif (bitcoins[x] == -1):
        buy(x, i)
    profits[i-100, x] = capital[x] + (bitcoins[x] * df["close"][i])
    x += 1

    if (df['macd'][i] > df['macd_signal'][i] + 4):
        while (bitcoins[x] < 1):
            buy(x, i)
    elif (df['macd'][i] < df['macd_signal'][i] - 4):
        while(bitcoins[x] > -1):
            sell(x, i)
    elif (bitcoins[x] == 1):
        sell(x, i)
    elif (bitcoins[x] == -1):
        buy(x, i)
    profits[i-100, x] = capital[x] + (bitcoins[x] * df["close"][i])
    x += 1

    if (df['macd'][i] > df['macd_signal'][i] + 7):
        while (bitcoins[x] < 1):
            buy(x, i)
    elif (df['macd'][i] < df['macd_signal'][i] - 7):
        while(bitcoins[x] > -1):
            sell(x, i)
    elif (bitcoins[x] == 1):
        sell(x, i)
    elif (bitcoins[x] == -1):
        buy(x, i)
    profits[i-100, x] = capital[x] + (bitcoins[x] * df["close"][i])
    x += 1

    if (df['macd'][i] > df['macd_signal'][i] + 10):
        while (bitcoins[x] < 1):
            buy(x, i)
    elif (df['macd'][i] < df['macd_signal'][i] - 10):
        while(bitcoins[x] > -1):
            sell(x, i)
    elif (bitcoins[x] == 1):
        sell(x, i)
    elif (bitcoins[x] == -1):
        buy(x, i)
    profits[i-100, x] = capital[x] + (bitcoins[x] * df["close"][i])
    x += 1

labels = ["0", "2", "4", "7", "10"]
for i in range(5):
    plt.plot(profits[:, i], label=labels[i])
plt.legend()
plt.show()
plt.plot(range(100, 1000), df["close"][100:])
plt.show()
