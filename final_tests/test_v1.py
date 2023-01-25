# imports
import ccxt
import talib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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
n_indicators = 5
aggression = 2  # Aggression variable from 1 to n_indicators/2
macd_change = 0
rsi_change = 0
bband_change = 0
stoch_change = 0
ema_change = 0

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


# Gathering data
source = "\\data_gathering\\store.csv"
ohlcv = pd.read_csv(os.getcwd() + source, header=None).to_numpy()
df = pd.DataFrame(
    ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
)

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

# Perform the trading

values = np.zeros(len(df) - 33)

for i in range(33, len(df)):
    if (
        np.sum(
            [
                (df["macd"].iloc[-1] > df["macd_signal"].iloc[-1] + macd_change),
                (df["rsi"].iloc[-1] > 50 + rsi_change),
                (
                    df["close"].iloc[-1]
                    > df["bollinger_middle"].iloc[-1] * (1 + bband_change)
                ),
                (df["stoch_k"].iloc[-1] > df["stoch_d"].iloc[-1] + stoch_change),
                (df["ema_12"].iloc[-1] > df["ema_26"].iloc[-1] * (1 + ema_change)),
            ]
        )
        > n_indicators - aggression
    ):
        # Buy signal
        if not long:
            long = True
            short = False
            while bitcoins < 1:
                buy(df["high"][i])
    elif (
        np.sum(
            [
                (df["macd"].iloc[-1] < df["macd_signal"].iloc[-1] - macd_change),
                (df["rsi"].iloc[-1] < 50 - rsi_change),
                (
                    df["close"].iloc[-1]
                    < df["bollinger_middle"].iloc[-1] * (1 - bband_change)
                ),
                (df["stoch_k"].iloc[-1] < df["stoch_d"].iloc[-1] - stoch_change),
                (df["ema_12"].iloc[-1] < df["ema_26"].iloc[-1] * (1 - ema_change)),
            ]
        )
        > n_indicators - aggression
    ):
        # Sell signal
        if not short:
            long = False
            short = True
            while bitcoins > -1:
                sell(df["low"][i])
    else:
        # Close positions
        if long:
            sell(df["low"][i])
        if short:
            buy(df["high"][i])
        long = False
        short = False

    values[i - 33] = capital + bitcoins * df["close"][i]

plt.plot(range(len(df) - 33), values)
plt.show()
