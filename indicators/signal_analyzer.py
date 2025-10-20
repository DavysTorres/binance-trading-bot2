import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil
import json
import os

# Módulos de optimización
class TechnicalOptimizer:
    """Optimizador de cálculos técnicos en lote"""
    
    @staticmethod
    def batch_calculate_indicators(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Calcula indicadores en lote para optimizar performance"""
        results = {}
        
        for symbol, data in data_dict.items():
            try:
                if len(data) < 10:
                    continue
                    
                # Calcular todos los indicadores de una vez
                results[symbol] = {
                    'rsi': TechnicalOptimizer._optimized_rsi(data),
                    'macd': TechnicalOptimizer._optimized_macd(data),
                    'atr': TechnicalOptimizer._optimized_atr(data),
                    'bollinger_bands': TechnicalOptimizer._optimized_bollinger_bands(data),
                    'sma': TechnicalOptimizer._optimized_sma(data),
                    'volume_profile': TechnicalOptimizer._optimized_volume_analysis(data),
                    'momentum': TechnicalOptimizer._optimized_momentum(data)
                }
            except Exception as e:
                logging.error(f"Error calculando indicadores para {symbol}: {e}")
        
        return results
    
    @staticmethod
    def _optimized_rsi(data: pd.DataFrame, period: int = 14) -> float:
        """RSI optimizado usando vectorización"""
        try:
            if len(data) < period:
                return 50.0
                
            close_prices = data['close'].values
            deltas = np.diff(close_prices)
            
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = pd.Series(gains).rolling(period-1).mean().iloc[-1]
            avg_loss = pd.Series(losses).rolling(period-1).mean().iloc[-1]
            
            if avg_loss == 0:
                return 100.0 if avg_gain > 0 else 50.0
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return min(max(rsi, 0), 100)
        except:
            return 50.0

    @staticmethod
    def _optimized_macd(data: pd.DataFrame) -> Dict:
        """MACD optimizado"""
        try:
            close_prices = data['close'].values
            
            # Usar pandas para cálculos EWM más eficientes
            close_series = pd.Series(close_prices)
            exp1 = close_series.ewm(span=12, adjust=False).mean()
            exp2 = close_series.ewm(span=26, adjust=False).mean()
            macd_line = exp1 - exp2
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            macd_histogram = macd_line - macd_signal
            
            return {
                'line': macd_line.iloc[-1] if not macd_line.empty else 0,
                'signal': macd_signal.iloc[-1] if not macd_signal.empty else 0,
                'histogram': macd_histogram.iloc[-1] if not macd_histogram.empty else 0,
                'crossed': macd_line.iloc[-1] > macd_signal.iloc[-1] if len(macd_line) > 1 else False
            }
        except:
            return {'line': 0, 'signal': 0, 'histogram': 0, 'crossed': False}
    
    @staticmethod
    def _optimized_atr(data: pd.DataFrame, period: int = 14) -> float:
        """ATR optimizado"""
        try:
            if len(data) < period:
                return 0.0
                
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Calcular True Range de manera vectorizada
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            
            true_range = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = np.mean(true_range[-period:]) if len(true_range) >= period else np.mean(true_range)
            
            return atr
        except:
            return 0.0
    
    @staticmethod
    def _optimized_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: int = 2) -> Dict:
        """Bandas de Bollinger optimizadas"""
        try:
            if len(data) < period:
                return {'upper': 0, 'lower': 0, 'position': 0.5}
                
            close_prices = data['close'].values
            sma = np.mean(close_prices[-period:])
            std = np.std(close_prices[-period:])
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            current_price = close_prices[-1]
            
            # Posición relativa en las bandas (0-1)
            if upper_band != lower_band:
                position = (current_price - lower_band) / (upper_band - lower_band)
            else:
                position = 0.5
                
            return {
                'upper': upper_band,
                'lower': lower_band,
                'sma': sma,
                'position': max(0, min(1, position)),
                'width': (upper_band - lower_band) / sma if sma > 0 else 0
            }
        except:
            return {'upper': 0, 'lower': 0, 'position': 0.5, 'width': 0}
    
    @staticmethod
    def _optimized_sma(data: pd.DataFrame) -> Dict:
        """Medias móviles optimizadas"""
        try:
            close_prices = data['close'].values
            
            if len(close_prices) >= 20:
                sma_20 = np.mean(close_prices[-20:])
                sma_50 = np.mean(close_prices[-50:]) if len(close_prices) >= 50 else sma_20
            else:
                sma_20 = np.mean(close_prices)
                sma_50 = sma_20
            
            trend = 'BULLISH' if sma_20 > sma_50 else 'BEARISH'
            distance = abs(sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0
            
            return {
                'sma_20': sma_20,
                'sma_50': sma_50,
                'trend': trend,
                'distance': distance
            }
        except:
            return {'sma_20': 0, 'sma_50': 0, 'trend': 'NEUTRAL', 'distance': 0}
    
    @staticmethod
    def _optimized_volume_analysis(data: pd.DataFrame) -> Dict:
        """Análisis de volumen optimizado"""
        try:
            volumes = data['volume'].values
            
            if len(volumes) < 5:
                return {'current': 0, 'avg_20': 0, 'ratio': 1.0}
            
            current_volume = volumes[-1]
            avg_volume_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
            
            return {
                'current': current_volume,
                'avg_20': avg_volume_20,
                'ratio': volume_ratio,
                'spike': volume_ratio > 1.5
            }
        except:
            return {'current': 0, 'avg_20': 0, 'ratio': 1.0, 'spike': False}
    
    @staticmethod
    def _optimized_momentum(data: pd.DataFrame) -> Dict:
        """Análisis de momentum optimizado"""
        try:
            close_prices = data['close'].values
            
            if len(close_prices) < 5:
                return {'trend': 'NEUTRAL', 'strength': 0.5, 'acceleration': 0}
            
            # Momentum simple (últimos 5 periodos)
            price_change = (close_prices[-1] - close_prices[-5]) / close_prices[-5] if close_prices[-5] > 0 else 0
            
            # Fuerza de la tendencia
            positive_periods = sum(1 for i in range(1, 6) if close_prices[-i] > close_prices[-i-1]) if len(close_prices) > 6 else 3
            
            trend_strength = positive_periods / 5.0
            trend = 'BULLISH' if price_change > 0 else 'BEARISH'
            
            return {
                'trend': trend,
                'strength': trend_strength,
                'price_change': price_change,
                'acceleration': price_change * trend_strength
            }
        except:
            return {'trend': 'NEUTRAL', 'strength': 0.5, 'price_change': 0, 'acceleration': 0}


class MemoryManager:
    """Gestor inteligente de memoria y cache"""
    
    def __init__(self, max_memory_mb: int = 100, cache_ttl_minutes: int = 5):
        self.max_memory = max_memory_mb * 1024 * 1024  # Convertir a bytes
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.data_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_cleanup = datetime.now()
    
    def get_cached_calculation(self, key: str, calculation_func, *args, **kwargs):
        """Obtiene cálculo cacheado o lo calcula si no existe"""
        # Limpieza periódica del cache
        if datetime.now() - self.last_cleanup > timedelta(minutes=1):
            self._clean_old_cache()
        
        if key in self.data_cache:
            data, timestamp = self.data_cache[key]
            if datetime.now() - timestamp < self.cache_ttl:
                self.cache_hits += 1
                return data
        
        self.cache_misses += 1
        result = calculation_func(*args, **kwargs)
        
        # Verificar memoria antes de cachear
        if self._get_memory_usage() < self.max_memory:
            self.data_cache[key] = (result, datetime.now())
        
        return result
    
    def _get_memory_usage(self) -> int:
        """Obtiene uso actual de memoria"""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except:
            return 0
    
    def _clean_old_cache(self):
        """Limpia cache antiguo"""
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, (value, timestamp) in self.data_cache.items():
            if current_time - timestamp > self.cache_ttl:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.data_cache[key]
        
        self.last_cleanup = current_time
    
    def get_cache_stats(self) -> Dict:
        """Obtiene estadísticas del cache"""
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'size': len(self.data_cache),
            'hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            'memory_usage_mb': self._get_memory_usage() / (1024 * 1024)
        }


class HistoricalValidator:
    """Validador basado en patrones históricos"""
    
    def __init__(self, history_file: str = "signal_history.json"):
        self.history_file = history_file
        self.history_db = self._load_history()
        self.min_occurrences = 3
        self.similarity_threshold = 0.7
    
    def _load_history(self) -> Dict:
        """Carga el historial desde archivo"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Error cargando historial: {e}")
        
        return {}
    
    def _save_history(self):
        """Guarda el historial en archivo"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history_db, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Error guardando historial: {e}")
    
    def validate_signal(self, signal: Dict, market_conditions: Dict = None) -> Dict:
        """Valida señal contra patrones históricos"""
        validation_result = {
            'historical_confidence': 0.5,
            'similar_patterns': 0,
            'success_rate': 0.5,
            'risk_adjustment': 1.0,
            'validation_notes': [],
            'is_valid': True
        }
        
        try:
            # Buscar patrones similares
            similar_patterns = self._find_similar_patterns(signal, market_conditions or {})
            validation_result['similar_patterns'] = len(similar_patterns)
            
            if similar_patterns:
                # Calcular métricas de éxito
                success_rate = self._calculate_pattern_success_rate(similar_patterns)
                validation_result['success_rate'] = success_rate
                validation_result['historical_confidence'] = success_rate
                
                # Ajustar confianza basado en número de ocurrencias
                occurrence_confidence = min(len(similar_patterns) / 10.0, 1.0)
                validation_result['historical_confidence'] *= occurrence_confidence
                
                # Aplicar ajustes de riesgo
                if success_rate < 0.4:
                    validation_result['risk_adjustment'] = 0.7
                    validation_result['validation_notes'].append("Patrón histórico de bajo éxito")
                    validation_result['is_valid'] = False
                elif success_rate > 0.7:
                    validation_result['risk_adjustment'] = 1.2
                    validation_result['validation_notes'].append("Patrón histórico de alto éxito")
            
            # Guardar señal actual en historial (sin resultado aún)
            self._add_to_history(signal, market_conditions)
            
        except Exception as e:
            logging.error(f"Error en validación histórica: {e}")
        
        return validation_result
    
    def _find_similar_patterns(self, signal: Dict, market_conditions: Dict) -> List[Dict]:
        """Encuentra patrones históricos similares"""
        similar = []
        
        for pattern_id, historical_signal in self.history_db.items():
            if self._are_patterns_similar(signal, historical_signal, market_conditions):
                similar.append(historical_signal)
        
        return similar
    
    def _are_patterns_similar(self, signal1: Dict, signal2: Dict, market_conditions: Dict) -> bool:
        """Determina si dos patrones son similares"""
        try:
            similarity_score = 0
            
            # Comparar tipo de señal
            if signal1.get('type') == signal2.get('type'):
                similarity_score += 0.3
            
            # Comparar score compuesto (dentro de 15 puntos)
            score1 = signal1.get('composite_score', 0)
            score2 = signal2.get('composite_score', 0)
            score_diff = abs(score1 - score2)
            if score_diff < 15:
                similarity_score += 0.3
            
            # Comparar timeframes
            tf1 = set(signal1.get('timeframes', []))
            tf2 = set(signal2.get('timeframes', []))
            if tf1 and tf2:
                tf_similarity = len(tf1.intersection(tf2)) / len(tf1.union(tf2))
                similarity_score += tf_similarity * 0.2
            
            # Comparar condiciones de mercado
            market_similarity = self._compare_market_conditions(
                signal1.get('market_conditions', {}),
                signal2.get('market_conditions', {})
            )
            similarity_score += market_similarity * 0.2
            
            return similarity_score >= self.similarity_threshold
            
        except Exception:
            return False
    
    def _compare_market_conditions(self, cond1: Dict, cond2: Dict) -> float:
        """Compara condiciones de mercado"""
        similarity = 0.0
        factors = 0
        
        for key in ['volatility', 'trend', 'volume_condition']:
            if key in cond1 and key in cond2:
                if cond1[key] == cond2[key]:
                    similarity += 1.0
                factors += 1
        
        return similarity / max(factors, 1)
    
    def _calculate_pattern_success_rate(self, patterns: List[Dict]) -> float:
        """Calcula tasa de éxito para patrones similares"""
        if not patterns:
            return 0.5
        
        successful = 0
        total_with_result = 0
        
        for pattern in patterns:
            if 'result' in pattern:
                total_with_result += 1
                if pattern['result'] == 'SUCCESS':
                    successful += 1
        
        return successful / max(total_with_result, 1)
    
    def _add_to_history(self, signal: Dict, market_conditions: Dict):
        """Añade señal al historial"""
        try:
            signal_id = f"{signal['symbol']}_{signal['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            historical_record = {
                'symbol': signal['symbol'],
                'type': signal['type'],
                'composite_score': signal.get('composite_score', 0),
                'timestamp': datetime.now().isoformat(),
                'timeframes': list(signal.get('timeframe_signals', {}).keys()),
                'market_conditions': market_conditions,
                'indicators': self._extract_key_indicators(signal)
            }
            
            self.history_db[signal_id] = historical_record
            
            # Mantener solo los últimos 1000 registros
            if len(self.history_db) > 1000:
                oldest_keys = sorted(self.history_db.keys())[:100]
                for key in oldest_keys:
                    del self.history_db[key]
            
            self._save_history()
            
        except Exception as e:
            logging.error(f"Error añadiendo al historial: {e}")
    
    def _extract_key_indicators(self, signal: Dict) -> Dict:
        """Extrae indicadores clave para el historial"""
        indicators = {}
        try:
            for tf, tf_data in signal.get('timeframe_signals', {}).items():
                tf_indicators = tf_data.get('indicators', {})
                indicators[tf] = {
                    'rsi': tf_indicators.get('rsi'),
                    'trend': tf_indicators.get('trend'),
                    'strength': tf_indicators.get('strength', 0)
                }
        except:
            pass
        
        return indicators
    
    def update_signal_result(self, signal_id: str, result: str, profit: float = 0):
        """Actualiza el resultado de una señal en el historial"""
        try:
            if signal_id in self.history_db:
                self.history_db[signal_id]['result'] = result
                self.history_db[signal_id]['profit'] = profit
                self.history_db[signal_id]['result_timestamp'] = datetime.now().isoformat()
                self._save_history()
        except Exception as e:
            logging.error(f"Error actualizando resultado: {e}")


class EnhancedSignalAnalyzer:
    """
    Analizador de señales técnicas optimizado con:
    - Cálculos en lote
    - Gestión inteligente de memoria
    - Validación histórica
    - Pesos dinámicos
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuración de filtros
        self.min_volume_usdt = config.get('min_volume_usdt', 100000)
        self.min_volatility = config.get('min_volatility_atr', 0.002)
        self.required_timeframes = config.get('required_timeframes', ['5m', '15m'])
        
        # Configuración de pesos dinámicos
        self.base_weights = {
            'rsi': 0.25,
            'macd': 0.25,
            'trend': 0.30,
            'momentum': 0.20
        }
        
        # Configuración de performance
        self.enable_caching = config.get('enable_caching', True)
        self.quick_analysis_mode = config.get('quick_analysis_mode', True)
        self.max_workers = config.get('max_workers', 4)
        
        # Módulos de optimización
        self.technical_optimizer = TechnicalOptimizer()
        self.memory_manager = MemoryManager(
            max_memory_mb=config.get('max_memory_mb', 100),
            cache_ttl_minutes=config.get('cache_ttl_minutes', 5)
        )
        self.historical_validator = HistoricalValidator()
        
        # Estadísticas
        self.analysis_stats = {
            'total_analyses': 0,
            'signals_detected': 0,
            'cache_performance': {}
        }
    
    def analyze_market_signals(self, symbol_data: Dict) -> List[Dict]:
        """
        Analiza señales técnicas con filtros de calidad integrados y validación histórica
        Versión optimizada con cálculos en lote y cache
        """
        self.analysis_stats['total_analyses'] += 1
        signals = []
        
        if not symbol_data:
            return signals
        
        try:
            # Pre-calcular indicadores en lote para todos los símbolos
            self.logger.info("🔄 Calculando indicadores técnicos en lote...")
            batch_indicators = self.technical_optimizer.batch_calculate_indicators(symbol_data)
            
            # Procesar cada símbolo de manera optimizada
            for symbol, timeframes_data in symbol_data.items():
                symbol_signals = self._analyze_symbol_optimized(
                    symbol, timeframes_data, batch_indicators.get(symbol, {})
                )
                signals.extend(symbol_signals)
            
            # Actualizar estadísticas
            self.analysis_stats['signals_detected'] += len(signals)
            self.analysis_stats['cache_performance'] = self.memory_manager.get_cache_stats()
            
            self.logger.info(f"🔍 Análisis completado: {len(signals)} señales detectadas")
            self.logger.info(f"📊 Stats cache: {self.analysis_stats['cache_performance']}")
            
        except Exception as e:
            self.logger.error(f"Error en análisis de mercado: {e}")
        
        return signals
    
    def _analyze_symbol_optimized(self, symbol: str, timeframes_data: Dict, 
                                precalculated_indicators: Dict) -> List[Dict]:
        """Análisis optimizado por símbolo usando indicadores pre-calculados"""
        try:
            # Verificación rápida de filtros usando cache
            cache_key = f"{symbol}_filters"
            passes_filters = self.memory_manager.get_cached_calculation(
                cache_key, 
                self._check_all_filters, 
                timeframes_data
            )
            
            if not passes_filters:
                return []
            
            # Obtener métricas de riesgo
            risk_cache_key = f"{symbol}_risk"
            risk_metrics = self.memory_manager.get_cached_calculation(
                risk_cache_key,
                self._calculate_risk_metrics,
                timeframes_data
            )
            
            # Análisis multi-timeframe optimizado
            tf_signals = {}
            for timeframe, data in timeframes_data.items():
                tf_key = f"{symbol}_{timeframe}_analysis"
                tf_signal = self.memory_manager.get_cached_calculation(
                    tf_key,
                    self._quick_timeframe_analysis,
                    data, timeframe, risk_metrics, precalculated_indicators
                )
                if tf_signal:
                    tf_signals[timeframe] = tf_signal
            
            # Generar señal compuesta si hay suficiente confluencia
            if len(tf_signals) >= 2:
                composite_signal = self._generate_composite_signal(
                    symbol, tf_signals, risk_metrics, precalculated_indicators
                )
                
                if composite_signal:
                    # Validación histórica
                    market_conditions = self._get_market_conditions(risk_metrics)
                    historical_validation = self.historical_validator.validate_signal(
                        composite_signal, market_conditions
                    )
                    
                    # Aplicar validación histórica
                    if historical_validation['is_valid']:
                        composite_signal['historical_validation'] = historical_validation
                        composite_signal['adjusted_score'] = composite_signal.get('composite_score', 0) * historical_validation['risk_adjustment']
                        return [composite_signal]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error en análisis optimizado de {symbol}: {e}")
            return []
    
    def _check_all_filters(self, timeframes_data: Dict) -> bool:
        """Verifica todos los filtros de una vez de manera optimizada"""
        return (
            self._passes_volume_filter(timeframes_data) and
            self._passes_volatility_filter(timeframes_data) and
            self._passes_liquidity_filter(timeframes_data)
        )
    
    def _passes_volume_filter(self, timeframes_data: Dict) -> bool:
        """Filtro optimizado de volumen"""
        try:
            latest_data = next(iter(timeframes_data.values()))
            if len(latest_data) == 0:
                return False
                
            current_volume = latest_data['volume'].iloc[-1]
            avg_volume = latest_data['volume'].tail(20).mean()
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            min_volume_condition = current_volume >= self.min_volume_usdt
            volume_spike_condition = 0.5 <= volume_ratio <= 3.0  # Rango razonable
            
            return min_volume_condition and volume_spike_condition
            
        except Exception as e:
            self.logger.debug(f"Error en filtro de volumen: {e}")
            return False
    
    def _passes_volatility_filter(self, timeframes_data: Dict) -> bool:
        """Filtro optimizado de volatilidad"""
        try:
            for tf, data in timeframes_data.items():
                if len(data) < 14:
                    continue
                    
                atr = self._calculate_atr_optimized(data)
                current_atr = atr if atr > 0 else 0
                
                # Volatilidad mínima pero no excesiva
                if not (self.min_volatility <= current_atr <= 0.05):
                    return False
            return True
        except Exception as e:
            self.logger.debug(f"Error en filtro de volatilidad: {e}")
            return False
    
    def _passes_liquidity_filter(self, timeframes_data: Dict) -> bool:
        """Filtro adicional de liquidez"""
        try:
            latest_data = next(iter(timeframes_data.values()))
            if len(latest_data) < 5:
                return False
                
            # Verificar que haya movimiento de precio reciente
            recent_data = latest_data.tail(5)
            price_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].iloc[-1]
            
            return price_range >= 0.005  # Al menos 0.5% de movimiento
        except:
            return False
    
    def _calculate_atr_optimized(self, data: pd.DataFrame) -> float:
        """ATR optimizado para filtros"""
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            if len(high) < 2:
                return 0.0
                
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            
            true_range = np.maximum(np.maximum(tr1, tr2), tr3)
            return np.mean(true_range[-14:]) if len(true_range) >= 14 else np.mean(true_range)
        except:
            return 0.0
    
    def _calculate_risk_metrics(self, timeframes_data: Dict) -> Dict:
        """Calcula métricas de riesgo de manera optimizada"""
        try:
            risk_metrics = {}
            
            # Usar el timeframe más alto para cálculos de riesgo
            highest_tf = max(timeframes_data.keys(), key=lambda x: self._get_timeframe_minutes(x))
            data = timeframes_data[highest_tf]
            
            if len(data) < 20:
                return {}
            
            # Cálculos vectorizados para mejor performance
            close_prices = data['close'].values
            
            # Volatilidad histórica
            returns = np.diff(close_prices) / close_prices[:-1]
            volatility = np.std(returns) * np.sqrt(365) if len(returns) > 1 else 0
            
            # Drawdown reciente
            rolling_max = np.maximum.accumulate(close_prices)
            drawdowns = (close_prices - rolling_max) / rolling_max
            max_drawdown = np.min(drawdowns[-50:]) if len(drawdowns) >= 50 else np.min(drawdowns)
            
            # Bandas de Bollinger
            bb = self.technical_optimizer._optimized_bollinger_bands(data)
            
            risk_metrics = {
                'volatility': abs(volatility),
                'max_drawdown': max_drawdown,
                'atr': self.technical_optimizer._optimized_atr(data),
                'bb_position': bb.get('position', 0.5),
                'bb_width': bb.get('width', 0),
                'timeframe': highest_tf,
                'price_trend': 'BULLISH' if close_prices[-1] > close_prices[-10] else 'BEARISH'
            }
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculando métricas de riesgo: {e}")
            return {}
    
    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """Convierte timeframe a minutos"""
        tf_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
            '12h': 720, '1d': 1440
        }
        return tf_map.get(timeframe, 15)
    
    def _quick_timeframe_analysis(self, data: pd.DataFrame, timeframe: str, 
                                risk_metrics: Dict, precalculated_indicators: Dict) -> Optional[Dict]:
        """Análisis rápido de timeframe usando indicadores pre-calculados"""
        if len(data) < 5:
            return None
        
        try:
            current_price = data['close'].iloc[-1]
            
            # Usar indicadores pre-calculados cuando sea posible
            signal_info = {
                'timeframe': timeframe,
                'price': current_price,
                'timestamp': datetime.now(),
                'indicators': self._get_enhanced_indicators(data, precalculated_indicators)
            }
            
            # Cálculo rápido de fuerza de señal
            signal_strength = self._quick_signal_strength(signal_info['indicators'])
            signal_info['strength'] = signal_strength
            
            return signal_info if signal_strength > 0.2 else None
            
        except Exception as e:
            self.logger.debug(f"Error en análisis rápido de {timeframe}: {e}")
            return None
    
    def _get_enhanced_indicators(self, data: pd.DataFrame, precalculated_indicators: Dict) -> Dict:
        """Obtiene indicadores mejorados combinando pre-calculados y cálculos rápidos"""
        indicators = {}
        
        # Usar indicadores pre-calculados
        if precalculated_indicators:
            indicators.update({
                'rsi': precalculated_indicators.get('rsi', 50),
                'macd': precalculated_indicators.get('macd', {}),
                'atr': precalculated_indicators.get('atr', 0),
                'bollinger_bands': precalculated_indicators.get('bollinger_bands', {}),
                'sma': precalculated_indicators.get('sma', {}),
                'volume': precalculated_indicators.get('volume_profile', {}),
                'momentum': precalculated_indicators.get('momentum', {})
            })
        else:
            # Cálculos de respaldo
            indicators.update({
                'rsi': self._calculate_rsi_quick(data),
                'trend': self._quick_trend_analysis(data),
                'momentum': self._quick_momentum_analysis(data)
            })
        
        return indicators
    
    def _calculate_rsi_quick(self, data: pd.DataFrame, period: int = 14) -> float:
        """RSI rápido para respaldo"""
        try:
            if len(data) < period:
                return 50.0
                
            close_prices = data['close'].tail(period + 1).values
            deltas = np.diff(close_prices)
            
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                return 100.0 if avg_gain > 0 else 50.0
                
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        except:
            return 50.0
    
    def _quick_trend_analysis(self, data: pd.DataFrame) -> str:
        """Análisis rápido de tendencia"""
        if len(data) < 5:
            return 'NEUTRAL'
        
        try:
            short_ma = data['close'].tail(5).mean()
            long_ma = data['close'].tail(15).mean() if len(data) >= 15 else short_ma
            
            if short_ma > long_ma * 1.005:
                return 'BULLISH'
            elif short_ma < long_ma * 0.995:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
        except:
            return 'NEUTRAL'
    
    def _quick_momentum_analysis(self, data: pd.DataFrame) -> Dict:
        """Análisis rápido de momentum"""
        try:
            if len(data) < 3:
                return {'trend': 'NEUTRAL', 'strength': 0.5}
            
            recent_prices = data['close'].tail(5).values
            price_changes = np.diff(recent_prices)
            
            positive_changes = sum(1 for change in price_changes if change > 0)
            momentum_strength = positive_changes / len(price_changes)
            
            trend = 'BULLISH' if momentum_strength > 0.6 else 'BEARISH' if momentum_strength < 0.4 else 'NEUTRAL'
            
            return {
                'trend': trend,
                'strength': momentum_strength,
                'direction': 'UP' if price_changes[-1] > 0 else 'DOWN'
            }
        except:
            return {'trend': 'NEUTRAL', 'strength': 0.5}
    
    def _quick_signal_strength(self, indicators: Dict) -> float:
        """Cálculo rápido de fuerza de señal"""
        try:
            strength = 0.0
            
            # RSI
            rsi = indicators.get('rsi', 50)
            if 30 <= rsi <= 70:
                strength += 0.2
            elif (rsi < 30 or rsi > 70):
                strength += 0.3
            
            # MACD
            macd_data = indicators.get('macd', {})
            if macd_data.get('crossed', False):
                strength += 0.3
            
            # Tendencia
            trend = indicators.get('sma', {}).get('trend', 'NEUTRAL')
            if trend in ['BULLISH', 'BEARISH']:
                strength += 0.3
            
            # Momentum
            momentum = indicators.get('momentum', {})
            if momentum.get('strength', 0.5) > 0.7:
                strength += 0.2
            
            return min(strength, 1.0)
            
        except:
            return 0.5
    
    def _generate_composite_signal(self, symbol: str, tf_signals: Dict, 
                                 risk_metrics: Dict, precalculated_indicators: Dict) -> Optional[Dict]:
        """Genera señal compuesta optimizada"""
        try:
            # Calcular score compuesto
            total_strength = sum(signal['strength'] for signal in tf_signals.values())
            avg_strength = total_strength / len(tf_signals)
            
            # Verificar confluencia de tendencia
            trends = []
            for signal in tf_signals.values():
                trend = signal['indicators'].get('sma', {}).get('trend') or \
                       signal['indicators'].get('trend') or \
                       signal['indicators'].get('momentum', {}).get('trend')
                if trend:
                    trends.append(trend)
            
            if not trends:
                return None
                
            bull_count = trends.count('BULLISH')
            bear_count = trends.count('BEARISH')
            
            signal_type = 'LONG' if bull_count > bear_count else 'SHORT'
            trend_confluence = max(bull_count, bear_count) / len(trends)
            
            # Calcular ajuste de riesgo
            risk_adjustment = self._calculate_risk_adjustment(risk_metrics, signal_type)
            adjusted_strength = avg_strength * risk_adjustment
            
            # Umbrales dinámicos basados en condiciones de mercado
            min_strength = 0.4 if risk_metrics.get('volatility', 0) < 0.03 else 0.5
            min_confluence = 0.6
            
            # Solo generar señal si cumple con los criterios
            if (avg_strength >= min_strength and 
                trend_confluence >= min_confluence and 
                risk_adjustment >= 0.6):
                
                composite_score = avg_strength * 100
                adjusted_score = adjusted_strength * 100
                
                return {
                    'symbol': symbol,
                    'type': signal_type,
                    'composite_score': round(composite_score, 2),
                    'adjusted_score': round(adjusted_score, 2),
                    'trend_confluence': round(trend_confluence * 100, 2),
                    'risk_adjustment': round(risk_adjustment * 100, 2),
                    'timestamp': datetime.now(),
                    'timeframe_signals': tf_signals,
                    'risk_metrics': risk_metrics,
                    'timeframes': list(tf_signals.keys()),
                    'metadata': {
                        'volume_checked': True,
                        'volatility_checked': True,
                        'multi_tf_confirmed': True,
                        'risk_adjusted': True,
                        'optimized_analysis': True
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generando señal compuesta para {symbol}: {e}")
            return None
    
    def _calculate_risk_adjustment(self, risk_metrics: Dict, signal_type: str) -> float:
        """Calcula ajuste de riesgo optimizado"""
        risk_adjustment = 1.0
        
        try:
            # Ajustar según volatilidad
            volatility = risk_metrics.get('volatility', 0)
            if volatility > 0.08:  # Muy alta volatilidad
                risk_adjustment *= 0.6
            elif volatility > 0.05:  # Alta volatilidad
                risk_adjustment *= 0.8
            elif volatility < 0.01:  # Muy baja volatilidad
                risk_adjustment *= 0.9
                
            # Ajustar según drawdown
            max_drawdown = abs(risk_metrics.get('max_drawdown', 0))
            if max_drawdown > 0.15:  # Gran drawdown reciente
                risk_adjustment *= 0.7
            elif max_drawdown > 0.08:
                risk_adjustment *= 0.85
                
            # Ajustar según posición en Bandas de Bollinger
            bb_position = risk_metrics.get('bb_position', 0.5)
            if signal_type == 'LONG' and bb_position > 0.8:  # Sobrecompra
                risk_adjustment *= 0.7
            elif signal_type == 'SHORT' and bb_position < 0.2:  # Sobreventa
                risk_adjustment *= 0.7
            elif 0.3 <= bb_position <= 0.7:  # Zona neutra
                risk_adjustment *= 1.1
                
            # Ajustar según ancho de bandas (volatilidad)
            bb_width = risk_metrics.get('bb_width', 0)
            if bb_width > 0.1:  # Bandas muy anchas (alta volatilidad)
                risk_adjustment *= 0.8
            elif bb_width < 0.02:  # Bandas muy estrechas (baja volatilidad)
                risk_adjustment *= 0.9
                
        except Exception as e:
            self.logger.error(f"Error calculando ajuste de riesgo: {e}")
            
        return max(risk_adjustment, 0.3)  # No reducir más del 70%
    
    def _get_market_conditions(self, risk_metrics: Dict) -> Dict:
        """Obtiene condiciones de mercado para validación histórica"""
        return {
            'volatility': 'HIGH' if risk_metrics.get('volatility', 0) > 0.05 else 'LOW',
            'trend': risk_metrics.get('price_trend', 'NEUTRAL'),
            'bb_position': risk_metrics.get('bb_position', 0.5),
            'volume_condition': 'NORMAL'  # Se puede expandir con más datos
        }
    
    def get_analysis_stats(self) -> Dict:
        """Obtiene estadísticas del análisis"""
        return {
            **self.analysis_stats,
            'cache_efficiency': self.memory_manager.get_cache_stats(),
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup(self):
        """Limpia recursos"""
        self.memory_manager._clean_old_cache()

# Funciones de utilidad para compatibilidad
def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Función legacy para compatibilidad"""
    return TechnicalOptimizer._optimized_rsi(data, period)

def calculate_macd(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Función legacy para compatibilidad"""
    macd_data = TechnicalOptimizer._optimized_macd(data)
    return pd.Series([macd_data['line']]), pd.Series([macd_data['signal']])

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Función legacy para compatibilidad"""
    atr_value = TechnicalOptimizer._optimized_atr(data, period)
    return pd.Series([atr_value])

# Ejemplo de uso
if __name__ == "__main__":
    # Configuración de ejemplo
    config = {
        'min_volume_usdt': 100000,
        'min_volatility_atr': 0.002,
        'required_timeframes': ['5m', '15m'],
        'enable_caching': True,
        'quick_analysis_mode': True,
        'max_memory_mb': 100,
        'cache_ttl_minutes': 5
    }
    
    # Inicializar analyzer
    analyzer = EnhancedSignalAnalyzer(config)
    
    print("✅ EnhancedSignalAnalyzer cargado correctamente")
    print("📊 Características implementadas:")
    print("   - Cálculos técnicos en lote")
    print("   - Gestión inteligente de memoria")
    print("   - Validación histórica de patrones")
    print("   - Análisis multi-timeframe optimizado")
    print("   - Filtros avanzados de volumen y volatilidad")