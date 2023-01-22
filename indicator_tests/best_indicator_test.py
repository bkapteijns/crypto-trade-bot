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

print("Balance:\t", "BTC:", exchange.fetch_balance()[
      "BTC"], "\n\t\t", "USD:", exchange.fetch_balance()["USDT"])

long = False
short = False

capital = np.array([30000]*10)
bitcoins = np.zeros(10)

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


profits = np.zeros((900, 10))

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

    if (df['rsi'][i] > 50):
        while (bitcoins[x] < 1):
            buy(x, i)
    else:
        while(bitcoins[x] > -1):
            sell(x, i)
    profits[i-100, x] = capital[x] + (bitcoins[x] * df["close"][i])
    x += 1

    if (df['close'][i] > df['bollinger_middle'][i]):
        while (bitcoins[x] < 1):
            buy(x, i)
    else:
        while(bitcoins[x] > -1):
            sell(x, i)
    profits[i-100, x] = capital[x] + (bitcoins[x] * df["close"][i])
    x += 1

    if (df['stoch_k'][i] > df['stoch_d'][i]):
        while (bitcoins[x] < 1):
            buy(x, i)
    else:
        while(bitcoins[x] > -1):
            sell(x, i)
    profits[i-100, x] = capital[x] + (bitcoins[x] * df["close"][i])
    x += 1

    if (df['ema_12'][i] > df['ema_26'][i]):
        while (bitcoins[x] < 1):
            buy(x, i)
    else:
        while(bitcoins[x] > -1):
            sell(x, i)
    profits[i-100, x] = capital[x] + (bitcoins[x] * df["close"][i])
    x += 1

    if (df['adx'][i] > 25):
        while (bitcoins[x] < 1):
            buy(x, i)
    else:
        while(bitcoins[x] > -1):
            sell(x, i)
    profits[i-100, x] = capital[x] + (bitcoins[x] * df["close"][i])
    x += 1

    if (df['cci'][i] > 100):
        while (bitcoins[x] < 1):
            buy(x, i)
    else:
        while(bitcoins[x] > -1):
            sell(x, i)
    profits[i-100, x] = capital[x] + (bitcoins[x] * df["close"][i])
    x += 1

    if (df['mfi'][i] > 80):
        while (bitcoins[x] < 1):
            buy(x, i)
    else:
        while(bitcoins[x] > -1):
            sell(x, i)
    profits[i-100, x] = capital[x] + (bitcoins[x] * df["close"][i])
    x += 1

    if (df['obv'][i] > df['obv'][i-1]):
        while (bitcoins[x] < 1):
            buy(x, i)
    else:
        while(bitcoins[x] > -1):
            sell(x, i)
    profits[i-100, x] = capital[x] + (bitcoins[x] * df["close"][i])
    x += 1

    if (df['close'][i] > df['sar'][i]):
        while (bitcoins[x] < 1):
            buy(x, i)
    else:
        while(bitcoins[x] > -1):
            sell(x, i)
    profits[i-100, x] = capital[x] + (bitcoins[x] * df["close"][i])

labels = ["macd", "rsi", "bbands", "stoch",
          "ema", "adx", "cci", "mfi", "obv", "sar"]
for i in range(10):
    plt.plot(profits[:, i], label=labels[i])
plt.legend()
plt.show()
