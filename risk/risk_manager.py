# trading/risk_manager.py
import pandas as pd
import numpy as np
from typing import Dict

class AdvancedRiskManager:
    """Gestor avanzado de riesgo"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_per_trade = config.get('trading', {}).get('risk_per_trade', 0.02)
        self.max_position_size = config.get('trading', {}).get('max_position_size_usdt', 1000)
    
    def calculate_position_parameters(self, signal: Dict) -> Dict:
        """Calcula parámetros de posición basados en riesgo"""
        symbol = signal['symbol']
        signal_type = signal['type']
        
        # Calcular tamaño de posición basado en riesgo
        account_balance = 1000  # En producción, obtener de OrderManager
        risk_amount = account_balance * self.risk_per_trade
        
        # Calcular precio de entrada, SL y TP
        entry_price = self._calculate_entry_price(signal)
        stop_loss = self._calculate_stop_loss(signal, entry_price)
        take_profit = self._calculate_take_profit(signal, entry_price)
        
        # Calcular cantidad
        price_diff = abs(entry_price - stop_loss)
        if price_diff > 0:
            quantity = risk_amount / price_diff
            # Ajustar por tamaño máximo
            quantity = min(quantity, self.max_position_size / entry_price)
        else:
            quantity = 0
        
        return {
            'quantity': round(quantity, 6),
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': risk_amount
        }
    
    def _calculate_entry_price(self, signal: Dict) -> float:
        """Calcula precio de entrada óptimo"""
        # Usar precio actual de la señal
        timeframe_data = next(iter(signal.get('timeframe_signals', {}).values()))
        return timeframe_data.get('price', 0)
    
    def _calculate_stop_loss(self, signal: Dict, entry_price: float) -> float:
        """Calcula stop loss basado en ATR o niveles técnicos"""
        atr = signal.get('risk_metrics', {}).get('atr', 0)
        
        if signal['type'] == 'LONG':
            return entry_price - (atr * 1.5)
        else:  # SHORT
            return entry_price + (atr * 1.5)
    
    def _calculate_take_profit(self, signal: Dict, entry_price: float) -> float:
        """Calcula take profit con ratio riesgo:beneficio"""
        stop_loss = self._calculate_stop_loss(signal, entry_price)
        risk = abs(entry_price - stop_loss)
        
        # Ratio 1:2 riesgo:beneficio
        if signal['type'] == 'LONG':
            return entry_price + (risk * 2)
        else:  # SHORT
            return entry_price - (risk * 2)