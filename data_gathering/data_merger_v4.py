import talib
import pandas as pd
import numpy as np
import os

future = 20
past = 100


def change(old, new):
    return ((new - old) / old) * 100


def merge_data(stores):
    X = []
    y = []

    for source in stores:
        # Training the model
        ohlcv = pd.read_csv(
            os.path.join(os.getcwd(), "data_gathering", source), header=None
        ).to_numpy()
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Calculate indicators
        df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
            df["close"], fastperiod=12, slowperiod=26, signalperiod=9
        )
        df["rsi"] = talib.RSI(df["close"], timeperiod=14)
        (
            df["bollinger_upper"],
            df["bollinger_middle"],
            df["bollinger_lower"],
        ) = talib.BBANDS(df["close"], timeperiod=20)
        df["stoch_k"], df["stoch_d"] = talib.STOCH(
            df["high"],
            df["low"],
            df["close"],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3,
        )
        df["ema_12"] = talib.EMA(df["close"], timeperiod=12)
        df["ema_26"] = talib.EMA(df["close"], timeperiod=26)
        df["adx"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
        df["cci"] = talib.CCI(df["high"], df["low"], df["close"], timeperiod=14)
        df["mfi"] = talib.MFI(
            df["high"], df["low"], df["close"], df["volume"], timeperiod=14
        )
        df["obv"] = talib.OBV(df["close"], df["volume"])
        df["sar"] = talib.SAR(df["high"], df["low"], acceleration=0.02, maximum=0.2)

        df = df.to_numpy()
        # first 4 columns are timestamp, open, high, and low; some indicators use previous data to compute (so you get NANs)
        df = df[33:, 4:]

        for i in range(100, len(df)):
            if change(df[i, 0], np.mean(df[i + 1 : i + 1 + future, 0])) > 0:
                y.append(1)
            else:
                y.append(-1)

        if len(X) == 0:
            X = np.zeros([len(df) - 100, 100, 18])
            for i in range(len(df) - 100):
                X[i] = df[i : i + 100]

            X = X.reshape([len(df) - 100, 18 * 100])
        else:
            new_X = np.zeros([len(df) - 100, 100, 18])
            for i in range(len(df) - 100):
                new_X[i] = df[i : i + 100]

            new_X = new_X.reshape([len(df) - 100, 18 * 100])
            X = np.vstack((X, new_X))

    return {"X": (X), "y": (y)}
