# risk/risk_config.py

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from enum import Enum

class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXTREME = 4

class MarketRegime(Enum):
    NORMAL = "normal"
    VOLATILE = "volatile"
    CRISIS = "crisis"
    TRENDING = "trending"

class RiskConfig:
    """Configuración completa de parámetros de riesgo"""
    
    # Límites de exposición global
    MAX_PORTFOLIO_EXPOSURE = 0.85  # 85% del capital total
    MAX_EXPOSURE_PER_SYMBOL = 0.12  # 12% por símbolo
    MAX_CORRELATED_EXPOSURE = 0.25  # 25% en activos correlacionados
    MAX_LEVERAGE_OVERALL = 3.0  # Leverage máximo total
    
    # Límites de pérdida
    MAX_DAILY_LOSS = 0.015  # 1.5% pérdida diaria máxima
    MAX_WEEKLY_LOSS = 0.05   # 5% pérdida semanal máxima
    MAX_MONTHLY_LOSS = 0.10  # 10% pérdida mensual máxima
    MAX_TRADE_DRAWDOWN = 0.04  # 4% drawdown por operación
    MAX_CONSECUTIVE_LOSSES = 4  # Máximo 4 pérdidas consecutivas
    
    # Parámetros de posición
    FIXED_RISK_PER_TRADE = 0.008  # 0.8% de riesgo por operación
    USE_KELLY_CRITERION = True
    KELLY_FRACTION = 0.35  # Usar 35% del criterio de Kelly
    MIN_POSITION_SIZE = 10  # USD mínimo
    MAX_POSITION_SIZE = 50000  # USD máximo
    
    # Puntuaciones de riesgo
    MAX_RISK_SCORE = 8  # Puntuación máxima para aprobar operación
    MAX_EXECUTION_RISK_SCORE = 3
    MAX_MARKET_RISK_SCORE = 4
    
    # Estrategias específicas
    STRATEGY_WIN_RATES = {
        'momentum': 0.58,
        'mean_reversion': 0.52,
        'breakout': 0.48,
        'arbitrage': 0.68,
        'scalping': 0.45,
        'swing': 0.55
    }
    
    AVG_WIN_LOSS_RATIO = {
        'momentum': 1.4,
        'mean_reversion': 1.1,
        'breakout': 1.8,
        'arbitrage': 1.05,
        'scalping': 1.3,
        'swing': 1.6
    }
    
    # Volatility thresholds
    VOLATILITY_NORMAL = 0.02  # 2% daily volatility
    VOLATILITY_HIGH = 0.04    # 4% daily volatility
    VOLATILITY_EXTREME = 0.08 # 8% daily volatility
    
    # Correlation thresholds
    HIGH_CORRELATION = 0.7
    MEDIUM_CORRELATION = 0.4
    
    # Time periods for analysis
    VOLATILITY_LOOKBACK = 30  # days
    CORRELATION_LOOKBACK = 90  # days