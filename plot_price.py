import pandas as pd
import matplotlib.pyplot as plt
import os

source = "\\data_gathering\\store.csv"
ohlcv = pd.read_csv(os.getcwd() + source, header=None)

prices = ohlcv[4]

plt.plot(range(len(ohlcv)), prices)
plt.show()
