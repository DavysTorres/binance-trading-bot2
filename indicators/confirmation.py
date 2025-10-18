import pandas as pd
import numpy as np

# ======================================================
# 📈 MACD (Moving Average Convergence Divergence)
# ======================================================
def calcular_macd(df: pd.DataFrame, short: int = 12, long: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calcula el indicador MACD:
    - MACD = EMA12 - EMA26
    - Signal = EMA9 del MACD
    Devuelve df con columnas ['MACD', 'Signal_MACD', 'Hist_MACD'].
    """
    df["EMA12"] = df["close"].ewm(span=short, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=long, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal_MACD"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["Hist_MACD"] = df["MACD"] - df["Signal_MACD"]
    return df


# ======================================================
# 📊 ADX (Average Directional Index)
# ======================================================
def calcular_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calcula el ADX (fuerza de tendencia).
    Retorna df con columnas: ['+DI', '-DI', 'ADX']
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    df["+DM"] = high.diff()
    df["-DM"] = -low.diff()

    df["+DM"] = np.where((df["+DM"] > df["-DM"]) & (df["+DM"] > 0), df["+DM"], 0.0)
    df["-DM"] = np.where((df["-DM"] > df["+DM"]) & (df["-DM"] > 0), df["-DM"], 0.0)

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(window=period).mean()

    df["+DI"] = 100 * (df["+DM"] / df["ATR"])
    df["-DI"] = 100 * (df["-DM"] / df["ATR"])
    df["DX"] = (abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])) * 100
    df["ADX"] = df["DX"].rolling(window=period).mean()

    return df


# ======================================================
# 🔁 Stochastic RSI
# ======================================================
def calcular_stoch_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calcula el Stochastic RSI (oscila entre 0 y 100)
    Ideal para medir sobrecompra/sobreventa.
    """
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    min_rsi = rsi.rolling(period).min()
    max_rsi = rsi.rolling(period).max()
    stoch_rsi = 100 * (rsi - min_rsi) / (max_rsi - min_rsi + 1e-10)

    df["RSI_14"] = rsi
    df["StochRSI"] = stoch_rsi
    return df


# ======================================================
# 📉 CCI (Commodity Channel Index)
# ======================================================
def calcular_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Calcula el Commodity Channel Index (CCI).
    Valores extremos:
      > +100 → sobrecompra
      < -100 → sobreventa
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df["CCI"] = (tp - sma) / (0.015 * mad)
    return df


# ======================================================
# 📊 Williams %R
# ======================================================
def calcular_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calcula el indicador Williams %R (de -100 a 0)
    -100 = sobreventa
     0 = sobrecompra
    """
    high_max = df["high"].rolling(window=period).max()
    low_min = df["low"].rolling(window=period).min()
    df["WilliamsR"] = -100 * (high_max - df["close"]) / (high_max - low_min + 1e-10)
    return df


# ======================================================
# 🧠 Confirmador Compuesto
# ======================================================
def aplicar_indicadores_confirmacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todos los indicadores de confirmación técnica (MACD, ADX, StochRSI, CCI, Williams%R)
    y devuelve el DataFrame enriquecido.
    """
    df = calcular_macd(df)
    df = calcular_adx(df)
    df = calcular_stoch_rsi(df)
    df = calcular_cci(df)
    df = calcular_williams_r(df)
    return df


# ======================================================
# ⚙️ Score de Confirmación
# ======================================================
def calcular_confianza_tecnica(df: pd.DataFrame) -> float:
    """
    Calcula un score (0-1) de confianza técnica basado en:
      - ADX > 25 (tendencia fuerte)
      - MACD > Signal
      - StochRSI > 80 o < 20 (momentum extremo)
      - CCI > 100 o < -100
      - WilliamsR < -80 (sobreventa) o > -20 (sobrecompra)
    """
    if df is None or len(df) < 2:
        return 0.0

    ultima = df.iloc[-1]
    score = 0.5  # base neutra

    # ADX
    if ultima.get("ADX", 0) > 25:
        score += 0.1

    # MACD cruzado
    if ultima.get("MACD", 0) > ultima.get("Signal_MACD", 0):
        score += 0.1

    # StochRSI momentum
    stoch = ultima.get("StochRSI", 50)
    if stoch < 20 or stoch > 80:
        score += 0.05

    # CCI extremo
    cci = ultima.get("CCI", 0)
    if abs(cci) > 100:
        score += 0.05

    # Williams %R
    willr = ultima.get("WilliamsR", -50)
    if willr < -80 or willr > -20:
        score += 0.05

    return min(1.0, round(score, 3))
