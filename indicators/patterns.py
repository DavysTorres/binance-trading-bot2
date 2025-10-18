# indicators/patterns.py
import pandas as pd
from datetime import datetime

# -----------------------------
# 🔍 Patrones individuales
# -----------------------------
def es_martillo(df: pd.DataFrame, n: int = -1) -> bool:
    """Detecta un patrón Hammer (reversión alcista)."""
    if len(df) < abs(n):
        return False

    vela = df.iloc[n]
    cuerpo = abs(vela["close"] - vela["open"])
    mecha_inferior = min(vela["open"], vela["close"]) - vela["low"]
    mecha_superior = vela["high"] - max(vela["open"], vela["close"])

    return (
        cuerpo > 0
        and mecha_inferior >= cuerpo * 2
        and mecha_superior <= cuerpo * 0.3
        and vela["close"] > vela["open"]
    )


def es_martillo_invertido(df: pd.DataFrame, n: int = -1) -> bool:
    """Detecta un patrón Inverted Hammer (reversión bajista)."""
    if len(df) < abs(n):
        return False

    vela = df.iloc[n]
    cuerpo = abs(vela["close"] - vela["open"])
    mecha_inferior = min(vela["open"], vela["close"]) - vela["low"]
    mecha_superior = vela["high"] - max(vela["open"], vela["close"])

    return (
        cuerpo > 0
        and mecha_superior >= cuerpo * 2
        and mecha_inferior <= cuerpo * 0.3
        and vela["close"] < vela["open"]
    )


def es_engulfing_alcista(df: pd.DataFrame) -> bool:
    """Detecta un patrón Bullish Engulfing."""
    if len(df) < 2:
        return False

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    return (
        prev["close"] < prev["open"]
        and curr["close"] > curr["open"]
        and curr["open"] < prev["close"]
        and curr["close"] > prev["open"]
    )


def es_engulfing_bajista(df: pd.DataFrame) -> bool:
    """Detecta un patrón Bearish Engulfing."""
    if len(df) < 2:
        return False

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    return (
        prev["close"] > prev["open"]
        and curr["close"] < curr["open"]
        and curr["open"] > prev["close"]
        and curr["close"] < prev["open"]
    )

# -----------------------------
# 🧠 Detección general
# -----------------------------
def detectar_patron_general(df: pd.DataFrame) -> str:
    """
    Devuelve el patrón dominante más reciente.
    Retorna: 'hammer', 'inverted_hammer', 'bullish_engulfing', 'bearish_engulfing', 'none'
    """
    if es_martillo(df):
        return "hammer"
    elif es_martillo_invertido(df):
        return "inverted_hammer"
    elif es_engulfing_alcista(df):
        return "bullish_engulfing"
    elif es_engulfing_bajista(df):
        return "bearish_engulfing"
    else:
        return "none"

# -----------------------------
# 🧩 Multi–timeframe y confianza
# -----------------------------
def evaluar_patron_multi(df_base: pd.DataFrame, df_superior: pd.DataFrame = None):
    """
    Evalúa el patrón actual en contexto multi-timeframe.
    Devuelve un dict con:
      {
        "patron": "hammer",
        "tendencia_superior": "alcista",
        "pattern_confidence": 0.85
      }
    """

    patron = detectar_patron_general(df_base)
    tendencia_superior = "indefinida"
    pattern_confidence = 0.0

    # Detectar tendencia general (EMA check simple)
    if "EMA_12" in df_base.columns and "EMA_26" in df_base.columns:
        if df_base["EMA_12"].iloc[-1] > df_base["EMA_26"].iloc[-1]:
            tendencia_superior = "alcista"
        elif df_base["EMA_12"].iloc[-1] < df_base["EMA_26"].iloc[-1]:
            tendencia_superior = "bajista"
        else:
            tendencia_superior = "lateral"

    # Si hay timeframe superior, usarlo para refuerzo
    if df_superior is not None and "EMA_12" in df_superior.columns:
        ema12, ema26 = df_superior["EMA_12"].iloc[-1], df_superior["EMA_26"].iloc[-1]
        if ema12 > ema26:
            tendencia_superior = "alcista"
        elif ema12 < ema26:
            tendencia_superior = "bajista"

    # Calcular confianza del patrón
    if patron == "hammer":
        pattern_confidence = 0.8 if tendencia_superior == "alcista" else 0.6
    elif patron == "inverted_hammer":
        pattern_confidence = 0.8 if tendencia_superior == "bajista" else 0.6
    elif patron == "bullish_engulfing":
        pattern_confidence = 0.9 if tendencia_superior == "alcista" else 0.7
    elif patron == "bearish_engulfing":
        pattern_confidence = 0.9 if tendencia_superior == "bajista" else 0.7
    else:
        pattern_confidence = 0.0

    return {
        "patron": patron,
        "tendencia_superior": tendencia_superior,
        "pattern_confidence": round(pattern_confidence, 2)
    }
