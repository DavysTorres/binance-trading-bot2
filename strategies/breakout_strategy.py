# strategies/breakout_strategy.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from .base_strategy import TradingStrategy

class BreakoutStrategy(TradingStrategy):
    """
    Estrategia de breakout que identifica rupturas de niveles
    clave con confirmación de volumen
    """
    
    def __init__(self, config: Dict):
        super().__init__("breakout", config)
        self.consolidation_period = config.get('consolidation_period', 20)
        self.volume_multiplier = config.get('volume_multiplier', 2.0)
        self.breakout_threshold = config.get('breakout_threshold', 0.02)  # 2%
        self.confirmation_candles = config.get('confirmation_candles', 2)
        
    async def analyze(self, symbol: str, data: Dict) -> Dict:
        """Analiza posibles breakouts en múltiples timeframes"""
        try:
            timeframe_signals = {}
            best_confidence = 0
            best_action = 'HOLD'
            
            for timeframe in ['4h', '1d']:  # Timeframes más adecuados para breakouts
                if timeframe in data:
                    df = data[timeframe]
                    df_with_indicators = self.calculate_indicators(df)
                    signal_df = self.generate_signals(df_with_indicators)
                    
                    if not signal_df.empty:
                        latest_signal = signal_df.iloc[-1]
                        action = self._interpret_breakout_signal(latest_signal)
                        confidence = self._calculate_breakout_confidence(latest_signal)
                        
                        timeframe_signals[timeframe] = {
                            'action': action,
                            'confidence': confidence,
                            'breakout_type': latest_signal.get('breakout_type', 'none'),
                            'volume_confirmation': latest_signal.get('volume_confirmation', False)
                        }
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_action = action
            
            return {
                'action': best_action,
                'confidence': best_confidence,
                'timeframe_breakouts': timeframe_signals,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error en análisis de breakout: {e}")
            return {'action': 'HOLD', 'confidence': 0}
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores para detección de breakouts"""
        df = df.copy()
        
        # Identificar rango de consolidación
        df['range_high'] = df['high'].rolling(window=self.consolidation_period).max()
        df['range_low'] = df['low'].rolling(window=self.consolidation_period).min()
        df['range_width'] = (df['range_high'] - df['range_low']) / df['range_low']
        
        # Volumen en el rango
        df['range_volume_avg'] = df['volume'].rolling(window=self.consolidation_period).mean()
        
        # Detectar posibles breakouts
        df['resistance_breakout'] = df['close'] > df['range_high'].shift(1)
        df['support_breakout'] = df['close'] < df['range_low'].shift(1)
        
        # Confirmación de volumen
        df['volume_spike'] = df['volume'] > (df['range_volume_avg'] * self.volume_multiplier)
        
        # Strength del breakout
        df['breakout_strength'] = self._calculate_breakout_strength(df)
        
        # ATR para stop loss
        df['atr'] = self._calculate_atr(df)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera señales de breakout"""
        df = df.copy()
        
        # Señales de breakout alcista
        bullish_breakout = (
            df['resistance_breakout'] &
            df['volume_spike'] &
            (df['range_width'] > 0.05)  # Rango significativo
        )
        
        # Señales de breakout bajista  
        bearish_breakout = (
            df['support_breakout'] &
            df['volume_spike'] &
            (df['range_width'] > 0.05)
        )
        
        # Tipo de breakout
        df['breakout_type'] = 'none'
        df.loc[bullish_breakout, 'breakout_type'] = 'bullish'
        df.loc[bearish_breakout, 'breakout_type'] = 'bearish'
        
        # Confirmación con velas posteriores
        df['breakout_confirmed'] = self._confirm_breakout(df)
        
        # Señal final
        df['signal'] = 0
        df.loc[(df['breakout_type'] == 'bullish') & df['breakout_confirmed'], 'signal'] = 1
        df.loc[(df['breakout_type'] == 'bearish') & df['breakout_confirmed'], 'signal'] = -1
        
        # Calidad del breakout
        df['breakout_quality'] = self._assess_breakout_quality(df)
        
        return df
    
    def _calculate_breakout_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calcula fuerza del breakout"""
        strength = pd.Series(0.0, index=df.index)
        
        # Para breakouts alcistas
        bullish_strength = (
            ((df['close'] - df['range_high']) / df['range_high'] * 5) +  # Distancia del breakout
            (df['volume'] / df['range_volume_avg'] * 0.3) +              # Multiplicador de volume
            (df['range_width'] * 2)                                      # Ancho del rango
        )
        
        # Para breakouts bajistas
        bearish_strength = (
            ((df['range_low'] - df['close']) / df['range_low'] * 5) +    # Distancia del breakout
            (df['volume'] / df['range_volume_avg'] * 0.3) +              # Multiplicador de volume
            (df['range_width'] * 2)                                      # Ancho del rango
        )
        
        strength[df['resistance_breakout']] = bullish_strength[df['resistance_breakout']]
        strength[df['support_breakout']] = bearish_strength[df['support_breakout']]
        
        return strength
    
    def _confirm_breakout(self, df: pd.DataFrame) -> pd.Series:
        """Confirma breakout con velas posteriores"""
        confirmed = pd.Series(False, index=df.index)
        
        for i in range(len(df)):
            if df['breakout_type'].iloc[i] != 'none':
                # Verificar siguientes velas mantienen el breakout
                lookahead = min(self.confirmation_candles, len(df) - i - 1)
                if lookahead > 0:
                    if df['breakout_type'].iloc[i] == 'bullish':
                        # Precio se mantiene arriba del breakout
                        subsequent_closes = [df['close'].iloc[i + j] for j in range(1, lookahead + 1)]
                        resistance_level = df['range_high'].iloc[i]
                        confirmed.iloc[i] = all(close > resistance_level for close in subsequent_closes)
                    
                    elif df['breakout_type'].iloc[i] == 'bearish':
                        # Precio se mantiene debajo del breakout
                        subsequent_closes = [df['close'].iloc[i + j] for j in range(1, lookahead + 1)]
                        support_level = df['range_low'].iloc[i]
                        confirmed.iloc[i] = all(close < support_level for close in subsequent_closes)
        
        return confirmed
    
    def _assess_breakout_quality(self, df: pd.DataFrame) -> pd.Series:
        """Evalúa calidad del breakout"""
        quality = pd.Series(0.0, index=df.index)
        
        quality_factors = [
            df['breakout_strength'] * 0.4,           # Fuerza del breakout
            (df['volume'] / df['range_volume_avg']) * 0.3,  # Confirmación de volume
            df['range_width'] * 2 * 0.2,             # Significancia del rango
            df['breakout_confirmed'].astype(float) * 0.1  # Confirmación de precio
        ]
        
        quality = sum(quality_factors)
        return quality.clip(0, 1)
    
    def _interpret_breakout_signal(self, signal_row: pd.Series) -> str:
        """Interpreta señal de breakout"""
        if signal_row['signal'] == 1 and signal_row['breakout_quality'] > 0.6:
            return 'BUY'
        elif signal_row['signal'] == -1 and signal_row['breakout_quality'] > 0.6:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_breakout_confidence(self, signal_row: pd.Series) -> float:
        """Calcula confianza para señal de breakout"""
        base_confidence = signal_row.get('breakout_quality', 0)
        
        # Ajustes basados en factores adicionales
        adjustments = 0
        
        # Volume confirmation boost
        if signal_row.get('volume_spike', False):
            adjustments += 0.2
        
        # Strong breakout strength
        if signal_row.get('breakout_strength', 0) > 0.5:
            adjustments += 0.1
        
        # Wide range breakout
        if signal_row.get('range_width', 0) > 0.1:
            adjustments += 0.1
        
        final_confidence = min(1.0, base_confidence + adjustments)
        return final_confidence
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula Average True Range"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr