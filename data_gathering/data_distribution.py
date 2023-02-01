import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def change(old, new):
    return ((new - old) / old) * 100


thresholds = [0, 0.05, 0.10, 0.20]
from data_merger_v4 import merge_data

y = np.array(merge_data()["y"])
distributions = [np.mean(y == -1), np.mean(y == 0), np.mean(y == 1)]

print(distributions)

plt.bar([-1, 0, 1], distributions)

plt.show()
