# strategies/multi_timeframe.py

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from sklearn.linear_model import LinearRegression

def analizar_curva(data):
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'] + ['']*6)
    df['close'] = df['close'].astype(float)

    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    df.dropna(inplace=True)

    X = np.arange(len(df[-20:])).reshape(-1, 1)
    y = df['close'].values[-20:]
    model = LinearRegression().fit(X, y)
    tendencia = model.coef_[0]

    #if df['rsi'].iloc[-1] < 30 and tendencia > 0:
    return True  # señal de compra
    return False
