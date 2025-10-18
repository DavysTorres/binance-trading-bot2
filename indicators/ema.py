# indicators/ema.py

import pandas as pd

def calcular_ema(df: pd.DataFrame, short: int = 12, long: int = 26, trend: int = 200) -> pd.DataFrame:
    """
    Calcula EMA de 12, 26 y 200 periodos y las agrega al DataFrame.
    """
    df["EMA_12"] = df["close"].ewm(span=short, adjust=False).mean()
    df["EMA_26"] = df["close"].ewm(span=long, adjust=False).mean()
    df["EMA_200"] = df["close"].ewm(span=trend, adjust=False).mean()
    return df

