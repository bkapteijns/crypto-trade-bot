import ccxt
import time
import numpy as np
import pandas as pd
import os

# Initialize the bot
exchange = ccxt.binance()
symbol = 'BTC/USDT'

# Create dataset
new_source = "\\data_gathering\\temp.csv"
old_source = "\\data_gathering\\store.csv"

while(True):
    ohlcv = exchange.fetch_ohlcv(symbol, "1m", limit=1000)

    np.savetxt(os.getcwd() + new_source, ohlcv, fmt="%.2f", delimiter=",")

    new_data = pd.read_csv(os.getcwd() + new_source, header=None).to_numpy()
    old_data = pd.read_csv(os.getcwd() + old_source, header=None).to_numpy()

    for i in range(len(new_data)):
        if (new_data[i, 0] == old_data[-1, 0]):
            old_data = np.append(old_data, new_data[i+1:, :], axis=0)
            break

    np.savetxt(os.getcwd() + old_source, old_data, fmt="%.2f", delimiter=",")

    print("new batch saved")

    time.sleep(50000)
