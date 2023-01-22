import ccxt
import numpy as np
import matplotlib.pyplot as plt

# Initialize the bot
exchange = ccxt.binance()
symbol = 'BTC/USDT'

prices = np.array(exchange.fetch_ohlcv(symbol, "1m", limit=1000))[:, 4]

plt.plot(range(1000), prices)
plt.show()
