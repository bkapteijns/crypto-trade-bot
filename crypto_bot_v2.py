import ccxt
import talib
import pandas as pd
import numpy as np
import time

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
aggression = 3  # Aggression variable from 1 to n_indicators/2

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


# Define trading strategy
while(True):  # Fetch historical data
    ohlcv = exchange.fetch_ohlcv(symbol, '1m')
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

    if np.sum([(df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] + 7), (df['rsi'].iloc[-1] > 50 + 2), (df['close'].iloc[-1] > df['bollinger_middle'].iloc[-1] * 1.0007), (df['stoch_k'].iloc[-1] > df['stoch_d'].iloc[-1] + 7), (df['adx'].iloc[-1] > 25 + 7)]) > n_indicators - aggression:
        # Buy signal
        if(not long):
            long = True
            short = False
            print("Bought at:", exchange.fetch_ticker(symbol)["ask"])
            while (bitcoins < 1):
                # exchange.create_market_buy_order(symbol, 1)
                buy(exchange.fetch_ticker(symbol)["ask"])
    elif np.sum([(df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] - 7), (df['rsi'].iloc[-1] < 50 - 2), (df['close'].iloc[-1] < df['bollinger_middle'].iloc[-1] * 0.9993), (df['stoch_k'].iloc[-1] < df['stoch_d'].iloc[-1] - 7), (df['adx'].iloc[-1] < 25 - 7)]) > n_indicators - aggression:
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
