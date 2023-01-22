import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def change(old, new):
    return ((new - old)/old) * 100


threshold = 0.08


source = "\\data_gathering\\store.csv"
ohlcv = pd.read_csv(os.getcwd() + source, header=None).to_numpy()
df = pd.DataFrame(
    ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
print(ohlcv.shape)
print(df.shape)


df = df.to_numpy()
# first 4 columns are timestamp, open, high, and low; some indicators use previous data to compute (so you get NANs)
df = df[33:, 4:]

y = np.zeros(len(df) - 133)
for i in range(100, len(df) - 33):
    if change(df[i, 0], np.mean(df[i+1:i+21, 0])) > threshold:
        y[i-100] = 1
    elif change(df[i, 0], np.mean(df[i+1:i+21, 0])) < 0-threshold:
        y[i-100] = -1

plt.bar([-1, 0, 1], [np.mean(y == -1), np.mean(y == 0), np.mean(y == 1)])
plt.show()
