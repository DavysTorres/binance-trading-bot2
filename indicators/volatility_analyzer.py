# indicators/volatility_analyzer.py

import pandas as pd
import numpy as np
import json


# Cargar configuración desde settings.json (solo una vez)
def cargar_config_volatilidad(config_path="config/settings.json"):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            min_std = config.get("min_std", 0.003)
            min_band_width = config.get("min_band_width", 0.01)
            min_atr_ratio = config.get("min_atr_ratio", 0.005)
            return min_std, min_band_width, min_atr_ratio
    except Exception as e:
        print(f"⚠️ Error al cargar configuración de volatilidad: {e}")
        # Retorna valores por defecto si hay error
        return 0.003, 0.01, 0.005
    
    
def calcular_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    return atr

def calcular_bollinger_band_width(df, period=20):
    close = df['close']
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    band_width = (upper_band - lower_band) / sma
    return band_width

def analizar_volatilidad(df, config_path="config/settings.json"):
    """
    Retorna True si hay buena volatilidad para operar.
    """
    if df is None or len(df) < 30:
        return False
    
    
    min_std, min_band_width, min_atr_ratio = cargar_config_volatilidad(config_path)

    # ATR
    atr = calcular_atr(df)
    atr_ratio = atr / df['close']

    # Band Width
    band_width = calcular_bollinger_band_width(df)

    # STD
    std_ = df['close'].rolling(window=20).std()

    ultima_std = std_.iloc[-1]
    ultima_bw = band_width.iloc[-1]
    ultima_atr = atr_ratio.iloc[-1]

    print(f"📈 Volatilidad: std={ultima_std:.5f}, BW={ultima_bw:.5f}, ATR={ultima_atr:.5f}")

    return (
        ultima_std > min_std and
        ultima_bw > min_band_width and
        ultima_atr > min_atr_ratio
    )
