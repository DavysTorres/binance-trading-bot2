# ===============================================================
# 📊 Pattern Context Analyzer
# ---------------------------------------------------------------
# Evalúa el contexto de las últimas velas para determinar
# si existe una tendencia alcista, bajista o lateral.
# Se usa como filtro de coherencia para los patrones de vela
# individuales detectados en patterns.py.
# ===============================================================

import pandas as pd
import numpy as np

# ===============================================================
# 🔍 Función principal
# ===============================================================
def analizar_contexto_velas(df: pd.DataFrame, lookback: int = 10) -> dict:
    """
    Evalúa el contexto de velas recientes para determinar la tendencia predominante.

    Retorna un dict con:
    {
        "trend_score": float (-1 a +1),
        "tendencia": "alcista" | "bajista" | "neutra",
        "velas_verdes": int,
        "velas_rojas": int,
        "vol_corr": float,
        "detalle": str
    }
    """

    if df is None or df.empty or len(df) < 3:
        return {
            "trend_score": 0,
            "tendencia": "indefinida",
            "velas_verdes": 0,
            "velas_rojas": 0,
            "vol_corr": 0,
            "detalle": "Datos insuficientes"
        }

    # Asegurar tamaño de ventana
    lookback = min(lookback, len(df))
    sub = df.tail(lookback).copy()

    closes = sub["close"].values
    opens = sub["open"].values
    highs = sub["high"].values
    lows = sub["low"].values
    volumes = sub["volume"].values

    # ============================================================
    # 1️⃣ Métricas básicas
    # ============================================================
    velas_verdes = int(np.sum(closes > opens))
    velas_rojas = int(np.sum(closes < opens))

    # Cambio porcentual entre la primera y última vela
    cambio_total = (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0

    # Media de cuerpo de velas y rango total
    cuerpos = np.abs(closes - opens)
    rangos = highs - lows
    body_ratio = np.mean(cuerpos / np.maximum(rangos, 1e-8))

    # ============================================================
    # 2️⃣ Evaluación de volumen y dirección
    # ============================================================
    try:
        vol_corr = np.corrcoef(range(len(volumes)), volumes)[0, 1]
    except Exception:
        vol_corr = 0

    # ============================================================
    # 3️⃣ Fuerza de tendencia (normalizada)
    # ============================================================
    fuerza_direccion = np.tanh(cambio_total * 10)  # normaliza entre -1 y +1
    tendencia_vol = 0.3 if vol_corr > 0 else -0.3 if vol_corr < 0 else 0

    # Score base: dirección + fuerza + volumen
    trend_score = fuerza_direccion + tendencia_vol
    trend_score = max(-1, min(1, trend_score))

    # ============================================================
    # 4️⃣ Determinación de tendencia textual
    # ============================================================
    if trend_score > 0.25 and velas_verdes >= velas_rojas:
        tendencia = "alcista"
    elif trend_score < -0.25 and velas_rojas > velas_verdes:
        tendencia = "bajista"
    else:
        tendencia = "neutra"

    # ============================================================
    # 5️⃣ Análisis de consolidación
    # ============================================================
    consolidacion_ratio = np.mean(cuerpos / np.maximum(rangos, 1e-8))
    if consolidacion_ratio < 0.25 and abs(trend_score) < 0.2:
        tendencia = "lateral"
        trend_score *= 0.5

    # ============================================================
    # 6️⃣ Detalle textual para logging
    # ============================================================
    detalle = (
        f"{velas_verdes}/{lookback} verdes, "
        f"ratio cuerpo={body_ratio:.2f}, "
        f"vol_corr={vol_corr:.2f}, "
        f"trend_score={trend_score:.2f}"
    )

    return {
        "trend_score": round(float(trend_score), 2),
        "tendencia": tendencia,
        "velas_verdes": velas_verdes,
        "velas_rojas": velas_rojas,
        "vol_corr": round(float(vol_corr), 3),
        "detalle": detalle
    }

# ===============================================================
# 🧪 Prueba rápida
# ===============================================================
if __name__ == "__main__":
    # Simulación básica
    datos = {
        "open": [1, 1.02, 1.05, 1.08, 1.1, 1.12, 1.15, 1.18, 1.2, 1.19],
        "close": [1.02, 1.05, 1.08, 1.1, 1.12, 1.15, 1.18, 1.2, 1.19, 1.17],
        "high": [1.03, 1.06, 1.09, 1.11, 1.13, 1.16, 1.19, 1.21, 1.2, 1.18],
        "low": [0.99, 1.01, 1.04, 1.07, 1.09, 1.12, 1.14, 1.17, 1.18, 1.16],
        "volume": [100, 120, 140, 180, 200, 220, 250, 240, 230, 200],
    }
    df = pd.DataFrame(datos)
    print(analizar_contexto_velas(df))
