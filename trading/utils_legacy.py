# trading/utils_legacy.py 
import numpy as np
import pandas as pd

def calcular_volatilidad(df):
    """
    Calcula la volatilidad (ATR aproximado) de forma robusta.
    Acepta tanto DataFrame como lista de velas de Binance.
    """
    try:
        # Si llega una lista de listas, convertirla
        if isinstance(df, list):
            if len(df) == 0:
                return 0.0
            df = pd.DataFrame(df, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)

        # Validación mínima
        if df is None or df.empty or "high" not in df or "low" not in df or "close" not in df:
            return 0.0

        # Calcular True Range
        df["prev_close"] = df["close"].shift(1)
        tr = np.maximum(df["high"] - df["low"],
                        np.maximum(abs(df["high"] - df["prev_close"]),
                                   abs(df["low"] - df["prev_close"])))

        # Promedio del TR (14 períodos)
        atr = tr.rolling(window=14).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else 0.0

    except Exception as e:
        print(f"⚠️ Error calculando volatilidad: {e}")
        return 0.0

