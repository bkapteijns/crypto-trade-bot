import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def change(old, new):
    return ((new - old)/old) * 100


thresholds = [0, 0.05, 0.10, 0.20]

distributions = np.zeros((4, 3))

for j, threshold in enumerate(thresholds):
    source = "\\data_gathering\\store.csv"
    ohlcv = pd.read_csv(os.getcwd() + source, header=None).to_numpy()
    df = pd.DataFrame(
        ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df = df.to_numpy()
    # first 4 columns are timestamp, open, high, and low; some indicators use previous data to compute (so you get NANs)
    df = df[33:, 4:]

    y = np.zeros(len(df) - 133)
    for i in range(100, len(df) - 33):
        if change(df[i, 0], np.mean(df[i+1:i+21, 0])) > threshold:
            y[i-100] = 1
        elif change(df[i, 0], np.mean(df[i+1:i+21, 0])) < 0-threshold:
            y[i-100] = -1
    distributions[j] = np.array(
        [np.mean(y == -1), np.mean(y == 0), np.mean(y == 1)])

print(distributions)

f, ax = plt.subplots(2, 2)

ax[0, 0].bar([-1, 0, 1], distributions[0])
ax[0, 1].bar([-1, 0, 1], distributions[1])
ax[1, 0].bar([-1, 0, 1], distributions[2])
ax[1, 1].bar([-1, 0, 1], distributions[3])

plt.show()
