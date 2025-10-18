# indicators/support_resistance.py
# ===============================================================
# 🧱 Detección simple de Soportes/Resistencias + SL/TP adaptativos
# ===============================================================

from __future__ import annotations
import pandas as pd
import numpy as np

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def detectar_sr(df: pd.DataFrame, window: int = 50, tol: float = 0.004):
    """
    Devuelve niveles S/R aproximados (promedios de clusters de máximos/mínimos locales).
    tol = porcentaje de tolerancia para agrupar niveles.
    """
    if len(df) < window:
        window = min(len(df), 30)
    if window < 10:
        return {"supports": [], "resistances": []}

    segment = df.tail(window).copy()
    highs = segment["high"]
    lows  = segment["low"]

    # extremos locales
    max_idx = (highs.shift(1) < highs) & (highs.shift(-1) < highs)
    min_idx = (lows.shift(1)  > lows)  & (lows.shift(-1)  > lows)

    max_levels = highs[max_idx].dropna().values.tolist()
    min_levels = lows[min_idx].dropna().values.tolist()
    if not max_levels and not min_levels:
        return {"supports": [], "resistances": []}

    def _cluster(levels):
        levels = sorted(levels)
        clusters = []
        for lv in levels:
            if not clusters:
                clusters.append([lv])
                continue
            center = np.mean(clusters[-1])
            if abs(lv - center) / max(center, 1e-9) <= tol:
                clusters[-1].append(lv)
            else:
                clusters.append([lv])
        return [float(np.mean(c)) for c in clusters]

    return {
        "supports": _cluster(min_levels),
        "resistances": _cluster(max_levels)
    }

def nivel_mas_cercano(price: float, levels: list[float]) -> float | None:
    if not levels:
        return None
    return float(sorted(levels, key=lambda x: abs(x - price))[0])

def proponer_sl_tp(df: pd.DataFrame, direction: str, rr: float = 1.5,
                   buffer_pct: float = 0.0015, atr_mult: float = 1.2):
    """
    SL/TP con S/R + fallback a ATR.
    - SL justo al otro lado del S/R cercano con pequeño buffer.
    - TP por múltiplo de RR del riesgo.
    """
    close = float(df["close"].iloc[-1])
    sr = detectar_sr(df, window=60, tol=0.004)
    atr = float(_atr(df, n=14).iloc[-1] or 0)

    if direction.lower() == "long":
        sup = nivel_mas_cercano(close, sr["supports"])
        if sup is not None and sup < close:
            sl = sup * (1 - buffer_pct)
        else:
            sl = close - (atr_mult * atr if atr > 0 else close * 0.01)
        risk = max(close - sl, close * 0.001)
        tp   = close + rr * risk
    else:
        res = nivel_mas_cercano(close, sr["resistances"])
        if res is not None and res > close:
            sl = res * (1 + buffer_pct)
        else:
            sl = close + (atr_mult * atr if atr > 0 else close * 0.01)
        risk = max(sl - close, close * 0.001)
        tp   = close - rr * risk

    return {
        "sl": float(round(sl, 8)),
        "tp": float(round(tp, 8)),
        "atr": atr,
        "sr": sr
    }
