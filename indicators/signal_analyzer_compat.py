# indicators/signal_analyzer_compat.py
import pandas as pd
from typing import Dict, Any

def detectar_senal(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Función de compatibilidad para mantener tu código existente
    """
    try:
        # Si la función original existe, usarla
        from indicators.signal_analyzer import detectar_senal as original_detectar
        return original_detectar(df)
    except ImportError:
        # Implementación básica de fallback
        return {
            "entrada": False,
            "tipo": "NONE", 
            "estrategia": "COMPATIBILITY",
            "confianza": 0.0
        }