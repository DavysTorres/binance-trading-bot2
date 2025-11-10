# strategies/momentum_strategy.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from strategies.base_strategy import TradingStrategy
from datetime import datetime

class MomentumStrategy(TradingStrategy):
    """
    Estrategia de momentum que combina múltiples timeframes
    y indicadores para detectar tendencias fuertes
    """
    
    def __init__(self, config: Dict):
        super().__init__("momentum", config)
        self.timeframes = config.get('timeframes', ['15m', '1h', '4h'])
        self.required_confidence = config.get('required_confidence', 0.6)
        
    async def analyze(self, symbol: str, data: Dict) -> Dict:
        """Analiza momentum en múltiples timeframes"""
        try:
            signals = {}
            timeframe_weights = {'15m': 0.2, '1h': 0.3, '4h': 0.5}
            total_weight = 0
            momentum_score = 0
            
            for timeframe in self.timeframes:
                if timeframe in data:
                    df = data[timeframe]
                    df_with_indicators = self.calculate_indicators(df)
                    signal_df = self.generate_signals(df_with_indicators)
                    
                    if not signal_df.empty:
                        latest_signal = signal_df.iloc[-1]
                        signals[timeframe] = latest_signal
                        
                        # Ponderar señal por timeframe
                        weight = timeframe_weights.get(timeframe, 0.2)
                        if latest_signal['momentum_signal'] == 1:
                            momentum_score += weight
                        elif latest_signal['momentum_signal'] == -1:
                            momentum_score -= weight
                        total_weight += weight
            
            if total_weight == 0:
                return {'action': 'HOLD', 'confidence': 0}
            
            # Normalizar score y calcular confianza
            normalized_score = momentum_score / total_weight
            confidence = abs(normalized_score)
            
            # Determinar acción
            if normalized_score > 0.1 and confidence >= self.required_confidence:
                action = 'BUY'
            elif normalized_score < -0.1 and confidence >= self.required_confidence:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            return {
                'action': action,
                'confidence': confidence,
                'momentum_score': normalized_score,
                'timeframe_signals': signals,
                'indicators': self._get_latest_indicators(data),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error en análisis de momentum: {e}")
            return {'action': 'HOLD', 'confidence': 0}
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores de momentum"""
        df = df.copy()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
        
        # Estocástico
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(
            df['high'], df['low'], df['close']
        )
        
        # Momentum simple
        df['price_momentum'] = df['close'] / df['close'].shift(5) - 1
        
        # ADX (Directional Movement)
        df['adx'] = self._calculate_adx(df['high'], df['low'], df['close'])
        
        # Ichimoku Cloud (simplificado)
        df = self._calculate_ichimoku(df)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera señales basadas en indicadores de momentum"""
        df = df.copy()
        
        # Señales individuales
        df['rsi_signal'] = self._get_rsi_signal(df['rsi'])
        df['macd_signal'] = self._get_macd_signal(df['macd'], df['macd_signal'])
        df['stoch_signal'] = self._get_stoch_signal(df['stoch_k'], df['stoch_d'])
        df['adx_signal'] = self._get_adx_signal(df['adx'])
        df['ichimoku_signal'] = self._get_ichimoku_signal(df)
        
        # Señal combinada de momentum
        signals = [
            df['rsi_signal'],
            df['macd_signal'], 
            df['stoch_signal'],
            df['adx_signal'],
            df['ichimoku_signal']
        ]
        
        # Promedio ponderado de señales
        weights = [0.2, 0.25, 0.15, 0.2, 0.2]
        df['momentum_signal'] = sum(s * w for s, w in zip(signals, weights))
        
        # Filtro de calidad
        df['signal_quality'] = self._calculate_signal_quality(df)
        df['final_signal'] = df['momentum_signal'] * df['signal_quality']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, 
                       fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, 
                            close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calcula Estocástico"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_k, stoch_d
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, period: int = 14) -> pd.Series:
        """Calcula Average Directional Index"""
        # Movimiento direccional
        high_diff = high.diff()
        low_diff = low.diff().abs()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Suavizado
        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula componentes de Ichimoku Cloud"""
        high = df['high']
        low = df['low']
        
        # Tenkan-sen (Conversion Line)
        period9_high = high.rolling(window=9).max()
        period9_low = low.rolling(window=9).min()
        df['tenkan_sen'] = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line)
        period26_high = high.rolling(window=26).max()
        period26_low = low.rolling(window=26).min()
        df['kijun_sen'] = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        period52_high = high.rolling(window=52).max()
        period52_low = low.rolling(window=52).min()
        df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        df['chikou_span'] = df['close'].shift(-26)
        
        return df
    
    def _get_rsi_signal(self, rsi: pd.Series) -> pd.Series:
        """Genera señal basada en RSI"""
        signal = pd.Series(0, index=rsi.index)
        signal[(rsi < 30) & (rsi.shift(1) >= 30)] = 1  # Oversold bounce
        signal[(rsi > 70) & (rsi.shift(1) <= 70)] = -1  # Overbought rejection
        return signal
    
    def _get_macd_signal(self, macd: pd.Series, macd_signal: pd.Series) -> pd.Series:
        """Genera señal basada en MACD"""
        signal = pd.Series(0, index=macd.index)
        signal[(macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))] = 1
        signal[(macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))] = -1
        return signal
    
    def _get_stoch_signal(self, stoch_k: pd.Series, stoch_d: pd.Series) -> pd.Series:
        """Genera señal basada en Estocástico"""
        signal = pd.Series(0, index=stoch_k.index)
        signal[(stoch_k < 20) & (stoch_d < 20) & (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))] = 1
        signal[(stoch_k > 80) & (stoch_d > 80) & (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))] = -1
        return signal
    
    def _get_adx_signal(self, adx: pd.Series, threshold: float = 25) -> pd.Series:
        """Genera señal basada en ADX"""
        signal = pd.Series(0, index=adx.index)
        signal[adx > threshold] = 1  # Tendencia fuerte
        signal[adx < 20] = -1  # Tendencia débil o rango
        return signal
    
    def _get_ichimoku_signal(self, df: pd.DataFrame) -> pd.Series:
        """Genera señal basada en Ichimoku"""
        signal = pd.Series(0, index=df.index)
        
        # Señal alcista: precio arriba de la nube, tenkan > kijun
        bullish_condition = (
            (df['close'] > df['senkou_span_a']) & 
            (df['close'] > df['senkou_span_b']) &
            (df['tenkan_sen'] > df['kijun_sen'])
        )
        
        # Señal bajista: precio debajo de la nube, tenkan < kijun  
        bearish_condition = (
            (df['close'] < df['senkou_span_a']) & 
            (df['close'] < df['senkou_span_b']) &
            (df['tenkan_sen'] < df['kijun_sen'])
        )
        
        signal[bullish_condition] = 1
        signal[bearish_condition] = -1
        
        return signal
    
    def _calculate_signal_quality(self, df: pd.DataFrame) -> pd.Series:
        """Calcula calidad de la señal basada en consistencia"""
        # Contar señales positivas en los indicadores
        signal_columns = ['rsi_signal', 'macd_signal', 'stoch_signal', 'adx_signal', 'ichimoku_signal']
        positive_signals = sum(df[col] > 0 for col in signal_columns)
        negative_signals = sum(df[col] < 0 for col in signal_columns)
        
        total_signals = len(signal_columns)
        quality = (abs(positive_signals - negative_signals) / total_signals)
        
        # Mejorar calidad si ADX muestra tendencia fuerte
        quality[df['adx'] > 25] *= 1.2
        quality[df['adx'] < 20] *= 0.8
        
        return quality.clip(0, 1)
    
    def _get_latest_indicators(self, data: Dict) -> Dict:
        """Obtiene los últimos valores de los indicadores"""
        latest_indicators = {}
        
        for timeframe in self.timeframes:
            if timeframe in data and not data[timeframe].empty:
                df = data[timeframe]
                latest = df.iloc[-1]
                latest_indicators[timeframe] = {
                    'rsi': latest.get('rsi', 0),
                    'macd': latest.get('macd', 0),
                    'macd_histogram': latest.get('macd_histogram', 0),
                    'stoch_k': latest.get('stoch_k', 0),
                    'adx': latest.get('adx', 0)
                }
        
        return latest_indicators