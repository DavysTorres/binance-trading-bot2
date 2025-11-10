# strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class TradingStrategy(ABC):
    """Clase abstracta base para todas las estrategias de trading"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"strategy.{name}")
        self.initialized = False
        self.performance_metrics = {}
        
    @abstractmethod
    async def analyze(self, symbol: str, data: Dict) -> Dict:
        """Analiza el símbolo y genera señales"""
        pass
    
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos para la estrategia"""
        pass
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera señales de compra/venta basadas en indicadores"""
        pass
    
    async def initialize(self):
        """Inicialización de la estrategia"""
        self.initialized = True
        self.logger.info(f"Estrategia {self.name} inicializada")
    
    def validate_signal(self, signal: Dict) -> bool:
        """Valida una señal antes de ejecución"""
        required_fields = ['symbol', 'action', 'confidence', 'timestamp']
        return all(field in signal for field in required_fields)
    
    def calculate_confidence(self, indicators: Dict) -> float:
        """Calcula confianza de la señal basada en múltiples factores"""
        return 0.0

class StrategyManager:
    """Gestiona múltiples estrategias y combina sus señales"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.strategies = {}
        self.signal_history = []
        self.logger = logging.getLogger("strategy.manager")
        
    def register_strategy(self, strategy: TradingStrategy):
        """Registra una estrategia en el manager"""
        self.strategies[strategy.name] = strategy
        self.logger.info(f"Estrategia registrada: {strategy.name}")
    
    async def get_combined_signals(self, symbol: str, data: Dict) -> Dict:
        """Combina señales de múltiples estrategias"""
        signals = {}
        total_confidence = 0
        strategy_count = 0
        
        for name, strategy in self.strategies.items():
            try:
                analysis = await strategy.analyze(symbol, data)
                if analysis.get('signal') and analysis['confidence'] > 0.3:
                    signals[name] = analysis
                    total_confidence += analysis['confidence']
                    strategy_count += 1
            except Exception as e:
                self.logger.error(f"Error en estrategia {name}: {e}")
        
        if not signals:
            return {'action': 'HOLD', 'confidence': 0, 'strategies': {}}
        
        # Calcular señal combinada
        avg_confidence = total_confidence / strategy_count
        buy_signals = sum(1 for s in signals.values() if s['action'] == 'BUY')
        sell_signals = sum(1 for s in signals.values() if s['action'] == 'SELL')
        
        if buy_signals > sell_signals:
            action = 'BUY'
        elif sell_signals > buy_signals:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'action': action,
            'confidence': avg_confidence,
            'strategy_count': strategy_count,
            'strategies': signals,
            'timestamp': datetime.now()
        }