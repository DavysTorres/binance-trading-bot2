# strategies/scalping_strategy.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from .base_strategy import TradingStrategy

class ScalpingStrategy(TradingStrategy):
    """
    Estrategia de scalping para timeframes cortos (1m, 5m)
    que busca pequeñas ineficiencias en el mercado
    """
    
    def __init__(self, config: Dict):
        super().__init__("scalping", config)
        self.fast_ema = config.get('fast_ema', 5)
        self.slow_ema = config.get('slow_ema', 15)
        self.rsi_period = config.get('rsi_period', 10)
        self.profit_target = config.get('profit_target', 0.003)  # 0.3%
        self.stop_loss = config.get('stop_loss', 0.002)  # 0.2%
        self.max_holding_time = config.get('max_holding_time', 10)  # minutos
        
    async def analyze(self, symbol: str, data: Dict) -> Dict:
        """Analiza oportunidades de scalping en timeframes cortos"""
        try:
            # Enfocarse en timeframes cortos para scalping
            short_timeframes = ['1m', '5m']
            signals = {}
            
            for timeframe in short_timeframes:
                if timeframe in data:
                    df = data[timeframe]
                    if len(df) < 20:  # Necesitamos suficientes datos
                        continue
                        
                    df_with_indicators = self.calculate_indicators(df)
                    signal_df = self.generate_signals(df_with_indicators)
                    
                    if not signal_df.empty:
                        latest_signal = signal_df.iloc[-1]
                        signals[timeframe] = latest_signal
            
            if not signals:
                return {'action': 'HOLD', 'confidence': 0}
            
            # Combinar señales de múltiples timeframes cortos
            combined_signal = self._combine_scalping_signals(signals)
            
            return {
                'action': combined_signal['action'],
                'confidence': combined_signal['confidence'],
                'timeframe_signals': signals,
                'profit_target': self.profit_target,
                'stop_loss': self.stop_loss,
                'max_holding_time': self.max_holding_time,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error en análisis de scalping: {e}")
            return {'action': 'HOLD', 'confidence': 0}
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores para scalping"""
        df = df.copy()
        
        # EMAs rápidas
        df['ema_fast'] = df['close'].ewm(span=self.fast_ema).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_ema).mean()
        
        # RSI rápido
        df['rsi_fast'] = self._calculate_rsi(df['close'], self.rsi_period)
        
        # Stochastic rápido
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(
            df['high'], df['low'], df['close'], 8, 3
        )
        
        # Volume analysis
        df['volume_ema'] = df['volume'].ewm(span=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ema']
        
        # Price momentum
        df['momentum'] = df['close'] / df['close'].shift(3) - 1
        
        # Bid-Ask spread estimation (usando high-low como proxy)
        df['spread_ratio'] = (df['high'] - df['low']) / df['close']
        
        # Order book imbalance (simulado)
        df['ob_imbalance'] = self._simulate_order_book_imbalance(df)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera señales de scalping"""
        df = df.copy()
        
        # Condiciones de compra para scalping
        buy_conditions = (
            (df['ema_fast'] > df['ema_slow']) &                    # EMA alignment
            (df['rsi_fast'] > 40) & (df['rsi_fast'] < 70) &       # RSI en zona óptima
            (df['stoch_k'] > df['stoch_d']) &                     # Stochastic crossover
            (df['volume_ratio'] > 1.2) &                          # Volume confirmation
            (df['momentum'] > 0) &                                # Momentum positivo
            (df['spread_ratio'] < 0.001) &                        # Spread tight
            (df['ob_imbalance'] > 0)                              # Order book favorable
        )
        
        # Condiciones de venta para scalping
        sell_conditions = (
            (df['ema_fast'] < df['ema_slow']) &                   # EMA alignment
            (df['rsi_fast'] < 60) & (df['rsi_fast'] > 30) &       # RSI en zona óptima
            (df['stoch_k'] < df['stoch_d']) &                     # Stochastic crossover
            (df['volume_ratio'] > 1.2) &                          # Volume confirmation
            (df['momentum'] < 0) &                                # Momentum negativo
            (df['spread_ratio'] < 0.001) &                        # Spread tight
            (df['ob_imbalance'] < 0)                              # Order book favorable
        )
        
        # Señales
        df['buy_signal'] = buy_conditions.astype(int)
        df['sell_signal'] = sell_conditions.astype(int)
        
        # Fuerza de la señal
        df['signal_strength'] = self._calculate_scalping_signal_strength(df)
        
        # Señal final (1 para buy, -1 para sell, 0 para hold)
        df['signal'] = 0
        df.loc[df['buy_signal'] == 1, 'signal'] = 1
        df.loc[df['sell_signal'] == 1, 'signal'] = -1
        
        # Filtrar señales débiles
        df.loc[df['signal_strength'] < 0.6, 'signal'] = 0
        
        return df
    
    def _calculate_scalping_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calcula fuerza de señal para scalping"""
        strength = pd.Series(0.0, index=df.index)
        
        # Factores para señales de compra
        buy_strength = (
            ((df['ema_fast'] - df['ema_slow']) / df['ema_slow'] * 1000) +  # EMA distance
            ((df['rsi_fast'] - 40) / 30 * 0.2) +                          # RSI position
            ((df['stoch_k'] - df['stoch_d']) * 0.3) +                     # Stochastic strength
            ((df['volume_ratio'] - 1) * 0.2) +                            # Volume strength
            (df['momentum'] * 100 * 0.2) +                                # Momentum
            (df['ob_imbalance'] * 0.1)                                    # Order book
        )
        
        # Factores para señales de venta
        sell_strength = (
            ((df['ema_slow'] - df['ema_fast']) / df['ema_fast'] * 1000) +  # EMA distance
            ((60 - df['rsi_fast']) / 30 * 0.2) +                          # RSI position
            ((df['stoch_d'] - df['stoch_k']) * 0.3) +                     # Stochastic strength
            ((df['volume_ratio'] - 1) * 0.2) +                            # Volume strength
            (abs(df['momentum']) * 100 * 0.2) +                           # Momentum
            (abs(df['ob_imbalance']) * 0.1)                               # Order book
        )
        
        strength[df['buy_signal'] == 1] = buy_strength[df['buy_signal'] == 1]
        strength[df['sell_signal'] == 1] = sell_strength[df['sell_signal'] == 1]
        
        return strength.clip(0, 1)
    
    def _combine_scalping_signals(self, signals: Dict) -> Dict:
        """Combina señales de múltiples timeframes de scalping"""
        if not signals:
            return {'action': 'HOLD', 'confidence': 0}
        
        buy_signals = 0
        sell_signals = 0
        total_confidence = 0
        signal_count = 0
        
        for timeframe, signal in signals.items():
            if signal['signal'] == 1:
                buy_signals += 1
                total_confidence += signal.get('signal_strength', 0)
                signal_count += 1
            elif signal['signal'] == -1:
                sell_signals += 1
                total_confidence += signal.get('signal_strength', 0)
                signal_count += 1
        
        if signal_count == 0:
            return {'action': 'HOLD', 'confidence': 0}
        
        avg_confidence = total_confidence / signal_count
        
        if buy_signals > sell_signals and avg_confidence > 0.6:
            return {'action': 'BUY', 'confidence': avg_confidence}
        elif sell_signals > buy_signals and avg_confidence > 0.6:
            return {'action': 'SELL', 'confidence': avg_confidence}
        else:
            return {'action': 'HOLD', 'confidence': avg_confidence}
    
    def _simulate_order_book_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Simula imbalance del order book usando price action"""
        # En producción, esto se reemplazaría con datos reales del order book
        imbalance = pd.Series(0.0, index=df.index)
        
        # Usar relación entre close, high, low para estimar presión
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['open'].iloc[i]:  # Vela alcista
                # Si el close está cerca del high, hay presión compradora
                if (df['high'].iloc[i] - df['close'].iloc[i]) / (df['high'].iloc[i] - df['low'].iloc[i]) < 0.3:
                    imbalance.iloc[i] = 0.5
            else:  # Vela bajista
                # Si el close está cerca del low, hay presión vendedora
                if (df['close'].iloc[i] - df['low'].iloc[i]) / (df['high'].iloc[i] - df['low'].iloc[i]) < 0.3:
                    imbalance.iloc[i] = -0.5
        
        return imbalance
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=period).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, 
                            close: pd.Series, k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
        """Calcula Estocástico"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_k, stoch_d