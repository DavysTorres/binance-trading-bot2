# indicators/rsi.py

import pandas as pd

def calcular_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calcula RSI de 14 periodos (por defecto) y lo agrega como columna 'RSI_14'
    """
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    return df
