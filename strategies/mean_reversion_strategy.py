# strategies/mean_reversion_strategy.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from .base_strategy import TradingStrategy

class MeanReversionStrategy(TradingStrategy):
    """
    Estrategia de reversión a la media que identifica activos
    sobrecomprados/sobrevendidos usando Bollinger Bands y Z-Score
    """
    
    def __init__(self, config: Dict):
        super().__init__("mean_reversion", config)
        self.lookback_period = config.get('lookback_period', 20)
        self.std_deviation = config.get('std_deviation', 2)
        self.rsi_period = config.get('rsi_period', 14)
        self.volume_threshold = config.get('volume_threshold', 1.5)
        
    async def analyze(self, symbol: str, data: Dict) -> Dict:
        """Analiza oportunidades de reversión a la media"""
        try:
            # Usar timeframe de 1h para mean reversion
            if '1h' not in data:
                return {'action': 'HOLD', 'confidence': 0}
            
            df = data['1h']
            df_with_indicators = self.calculate_indicators(df)
            signal_df = self.generate_signals(df_with_indicators)
            
            if signal_df.empty:
                return {'action': 'HOLD', 'confidence': 0}
            
            latest_signal = signal_df.iloc[-1]
            action = self._interpret_signal(latest_signal)
            confidence = self._calculate_confidence(latest_signal)
            
            return {
                'action': action,
                'confidence': confidence,
                'bollinger_position': latest_signal.get('bollinger_position', 0),
                'z_score': latest_signal.get('z_score', 0),
                'rsi': latest_signal.get('rsi', 50),
                'volume_spike': latest_signal.get('volume_spike', False),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error en análisis de mean reversion: {e}")
            return {'action': 'HOLD', 'confidence': 0}
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores para mean reversion"""
        df = df.copy()
        
        # Bollinger Bands
        df['sma'] = df['close'].rolling(window=self.lookback_period).mean()
        df['std'] = df['close'].rolling(window=self.lookback_period).std()
        df['bollinger_upper'] = df['sma'] + (df['std'] * self.std_deviation)
        df['bollinger_lower'] = df['sma'] - (df['std'] * self.std_deviation)
        
        # Posición en Bollinger Bands
        df['bollinger_position'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
        
        # Z-Score
        df['z_score'] = (df['close'] - df['sma']) / df['std']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_spike'] = df['volume_ratio'] > self.volume_threshold
        
        # Mean reversion momentum
        df['price_vs_sma'] = df['close'] / df['sma'] - 1
        
        # Support/Resistance levels
        df = self._calculate_support_resistance(df)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera señales de mean reversion"""
        df = df.copy()
        
        # Condiciones de compra (sobrevendido)
        buy_conditions = (
            (df['z_score'] < -self.std_deviation) |  # 2 std debajo de la media
            (df['bollinger_position'] < 0.1) |       # Cerca del bollinger lower
            (df['rsi'] < 30)                         # RSI oversold
        )
        
        # Condiciones de venta (sobrecomprado)
        sell_conditions = (
            (df['z_score'] > self.std_deviation) |   # 2 std arriba de la media
            (df['bollinger_position'] > 0.9) |       # Cerca del bollinger upper
            (df['rsi'] > 70)                         # RSI overbought
        )
        
        # Señales
        df['buy_signal'] = 0
        df['sell_signal'] = 0
        
        df.loc[buy_conditions, 'buy_signal'] = 1
        df.loc[sell_conditions, 'sell_signal'] = 1
        
        # Filtro de confirmación
        df['confirmed_buy'] = (
            df['buy_signal'] & 
            (df['volume_spike']) &  # Volume spike en dirección
            (df['price_vs_sma'].shift(1) < df['price_vs_sma'])  # Mejoría reciente
        )
        
        df['confirmed_sell'] = (
            df['sell_signal'] &
            (df['volume_spike']) &
            (df['price_vs_sma'].shift(1) > df['price_vs_sma'])
        )
        
        # Señal final
        df['signal'] = 0
        df.loc[df['confirmed_buy'], 'signal'] = 1
        df.loc[df['confirmed_sell'], 'signal'] = -1
        
        # Strength del signal
        df['signal_strength'] = self._calculate_signal_strength(df)
        
        return df
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula niveles de soporte y resistencia"""
        # Pivots simples
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['resistance1'] = 2 * df['pivot'] - df['low']
        df['support1'] = 2 * df['pivot'] - df['high']
        
        # Distancia a niveles clave
        df['distance_to_resistance'] = (df['resistance1'] - df['close']) / df['close']
        df['distance_to_support'] = (df['close'] - df['support1']) / df['close']
        
        return df
    
    def _calculate_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calcula fuerza de la señal basada en múltiples factores"""
        strength = pd.Series(0.0, index=df.index)
        
        # Factores para señales de compra
        buy_strength = (
            (abs(df['z_score']) * 0.3) +                    # Magnitud del z-score
            ((1 - df['bollinger_position']) * 0.3) +        # Posición en bollinger
            ((30 - df['rsi']) / 30 * 0.2) +                 # RSI oversold
            (df['volume_ratio'] * 0.1) +                    # Volume confirmation
            (df['distance_to_resistance'] * 10 * 0.1)       # Potencial upside
        )
        
        # Factores para señales de venta
        sell_strength = (
            (abs(df['z_score']) * 0.3) +                    # Magnitud del z-score
            (df['bollinger_position'] * 0.3) +              # Posición en bollinger
            ((df['rsi'] - 70) / 30 * 0.2) +                 # RSI overbought
            (df['volume_ratio'] * 0.1) +                    # Volume confirmation
            (df['distance_to_support'] * 10 * 0.1)          # Potencial downside
        )
        
        strength[df['signal'] == 1] = buy_strength[df['signal'] == 1]
        strength[df['signal'] == -1] = sell_strength[df['signal'] == -1]
        
        return strength.clip(0, 1)
    
    def _interpret_signal(self, signal_row: pd.Series) -> str:
        """Interpreta la señal numérica como acción"""
        if signal_row['signal'] == 1 and signal_row['signal_strength'] > 0.5:
            return 'BUY'
        elif signal_row['signal'] == -1 and signal_row['signal_strength'] > 0.5:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_confidence(self, signal_row: pd.Series) -> float:
        """Calcula confianza basada en fuerza de señal y confirmaciones"""
        base_confidence = signal_row.get('signal_strength', 0)
        
        # Aumentar confianza con confirmaciones
        confirmations = 0
        if signal_row.get('volume_spike', False):
            confirmations += 1
        if abs(signal_row.get('z_score', 0)) > 1.5:
            confirmations += 1
        if signal_row.get('rsi', 50) < 35 or signal_row.get('rsi', 50) > 65:
            confirmations += 1
        
        confidence_boost = confirmations * 0.1
        final_confidence = min(1.0, base_confidence + confidence_boost)
        
        return final_confidence
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))