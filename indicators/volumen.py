# indicators/volumen.py

import pandas as pd

def calcular_volumen_promedio(df: pd.DataFrame, periodo: int = 20) -> pd.Series:
    """
    Calcula volumen promedio móvil del período dado y devuelve la serie.
    """
    return df["volume"].rolling(window=periodo).mean()

def es_breakout_volumen(df: pd.DataFrame, factor: float = 1.5) -> bool:
    """
    Retorna True si el volumen actual supera el promedio * factor.
    """
    if "volumen_movil_20" not in df.columns:
        df["volumen_movil_20"] = df["volume"].rolling(window=20).mean()

    vol_actual = df["volume"].iloc[-1]
    vol_prom = df["volumen_movil_20"].iloc[-1]
    return vol_actual > vol_prom * factor

