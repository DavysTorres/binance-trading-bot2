# core/signal_reevaluator.py
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json
import os

@dataclass
class ReevaluationResult:
    """Resultado de la re-evaluación de una señal"""
    signal: Dict
    is_confirmed: bool
    confidence: float
    entry_quality: str  # EXCELLENT, GOOD, POOR, REJECTED
    adjustment_factors: Dict
    recommendation: str  # EXECUTE, WAIT, CANCEL
    reevaluation_timestamp: datetime

class OptimizedSignalReevaluator:
    """
    Re-evaluador optimizado para producción:
    - Verifica contexto de mercado en tiempo real
    - Evalúa calidad del punto de entrada
    - Confirma o descarta señales cada ~3 minutos
    - Integra con el análisis optimizado
    - Manejo robusto de errores y reintentos
    - Circuit breaker pattern para prevención de cascadas
    """

    def __init__(self, config: Dict, data_loader, max_retries: int = 3, retry_delay: float = 1.0):
        self.config = config
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__)

        # Configuración
        self.reeval_interval = config.get('reevaluation_interval_minutes', 3)
        self.min_confidence = config.get('min_confidence_threshold', 0.7)
        self.max_position_size = config.get('max_position_size_usdt', 1000)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Configuración de producción
        reeval_config = config.get('signal_reevaluator', {})
        perf_config = reeval_config.get('performance', {})
        
        self.circuit_breaker_enabled = perf_config.get('circuit_breaker_enabled', True)
        self.max_errors_per_minute = perf_config.get('max_errors_per_minute', 10)
        self.request_timeout = perf_config.get('request_timeout', 10.0)
        self.overall_timeout = perf_config.get('overall_timeout', 30.0)
        self.error_timestamps = []

        # Umbrales de calidad
        quality_config = reeval_config.get('quality_thresholds', {})
        self.quality_thresholds = {
            'excellent': quality_config.get('excellent', 0.85),
            'good': quality_config.get('good', 0.70),
            'poor': quality_config.get('poor', 0.60)
        }

        # Estadísticas
        self.stats = {
            'total_reevaluations': 0,
            'signals_confirmed': 0,
            'signals_rejected': 0,
            'last_reevaluation': None,
            'retry_count': 0,
            'error_count': 0,
            'circuit_breaker_triggers': 0,
            'timeout_errors': 0
        }

        # Cache para datos de mercado (evita llamadas duplicadas)
        self.market_data_cache = {}
        self.cache_ttl = timedelta(minutes=2)

        self.logger.info("🚀 SignalReevaluator inicializado para producción")

    async def reevaluate_signals_batch(self, signals: List[Dict]) -> List[ReevaluationResult]:
        """
        Re-evalúa un lote de señales de manera optimizada con agrupación por símbolo
        Incluye circuit breaker y manejo robusto de errores
        """
        # Verificar circuit breaker
        if self.circuit_breaker_enabled and self._is_circuit_open():
            self.stats['circuit_breaker_triggers'] += 1
            self.logger.warning("🚨 Circuit breaker abierto, omitiendo re-evaluación")
            return [self._create_rejected_result(s, "Circuit breaker activado - sistema en modo seguro") for s in signals]

        if not signals:
            self.logger.info("📭 No hay señales para re-evaluar")
            return []

        self.stats['total_reevaluations'] += len(signals)
        self.stats['last_reevaluation'] = datetime.now().isoformat()

        results = []
        self.logger.info(f"🔁 Re-evaluando lote de {len(signals)} señales...")

        try:
            # Usar timeout global para todo el proceso
            async with asyncio.timeout(self.overall_timeout):
                # Agrupar señales por símbolo para reducir llamadas a datos de mercado
                symbol_groups = {}
                for signal in signals:
                    symbol = signal['symbol']
                    if symbol not in symbol_groups:
                        symbol_groups[symbol] = []
                    symbol_groups[symbol].append(signal)

                # Procesar cada grupo de símbolos
                for symbol, symbol_signals in symbol_groups.items():
                    symbol_results = await self._process_symbol_group(symbol, symbol_signals)
                    results.extend(symbol_results)

        except asyncio.TimeoutError:
            self.stats['timeout_errors'] += 1
            self.logger.error(f"⏰ Timeout global de {self.overall_timeout}s alcanzado en re-evaluación")
            # Crear resultados de error para todas las señales
            results = [self._create_rejected_result(s, f"Timeout global de {self.overall_timeout}s") for s in signals]
            self._record_error()

        except Exception as e:
            self.logger.error(f"💥 Error crítico en re-evaluación por lotes: {e}")
            self._record_error()
            # Fallback seguro - rechazar todas las señales
            results = [self._create_rejected_result(s, f"Error del sistema: {str(e)}") for s in signals]

        # Log resumen y actualizar estadísticas
        self._log_reevaluation_summary(results)
        confirmed_count = sum(1 for r in results if r.is_confirmed)
        self.stats['signals_confirmed'] += confirmed_count
        self.stats['signals_rejected'] += (len(signals) - confirmed_count)

        # Limpiar cache antiguo
        self._clean_old_cache()

        return results

    async def _process_symbol_group(self, symbol: str, symbol_signals: List[Dict]) -> List[ReevaluationResult]:
        """Procesa un grupo de señales del mismo símbolo"""
        results = []

        try:
            # Obtener datos de mercado una vez por símbolo (con cache)
            current_data = await self._get_cached_market_data(symbol)
            if not current_data:
                error_msg = "Datos de mercado no disponibles después de reintentos"
                self.logger.warning(f"⚠️ {error_msg} para {symbol}")
                return [self._create_rejected_result(s, error_msg) for s in symbol_signals]

            # Analizar contexto de mercado una vez por símbolo
            market_context = await self._analyze_market_context(symbol, current_data)

            # Evaluar métricas de riesgo una vez por símbolo
            risk_metrics = self._evaluate_risk_metrics(symbol_signals[0], current_data)

            # Reevaluar cada señal del grupo
            for signal in symbol_signals:
                try:
                    result = await self._reevaluate_single_signal(
                        signal, current_data, market_context, risk_metrics
                    )
                    results.append(result)

                except Exception as e:
                    self.logger.error(f"❌ Error re-evaluando señal {signal.get('symbol')}: {e}")
                    error_result = self._create_rejected_result(signal, f"Error de procesamiento: {str(e)}")
                    results.append(error_result)
                    self.stats['error_count'] += 1

        except Exception as e:
            # Manejo de errores a nivel de símbolo
            self.logger.error(f"❌ Error procesando grupo de símbolo {symbol}: {e}")
            self._record_error()
            for signal in symbol_signals:
                error_result = self._create_rejected_result(signal, f"Error procesando símbolo: {str(e)}")
                results.append(error_result)

        return results

    async def _get_cached_market_data(self, symbol: str) -> Optional[Dict]:
        """Obtiene datos de mercado con cache y reintentos"""
        # Verificar cache primero
        cache_key = f"{symbol}_market_data"
        if cache_key in self.market_data_cache:
            data, timestamp = self.market_data_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                self.logger.debug(f"📦 Usando datos en cache para {symbol}")
                return data

        # Obtener datos frescos
        data = await self._get_current_market_data_with_retry(symbol)
        if data:
            self.market_data_cache[cache_key] = (data, datetime.now())
        return data

    async def _get_current_market_data_with_retry(self, symbol: str) -> Optional[Dict]:
        """Obtiene datos de mercado con reintentos y timeouts"""
        for attempt in range(self.max_retries):
            try:
                async with asyncio.timeout(self.request_timeout):
                    data = await self._get_current_market_data(symbol)
                    if data:
                        return data
                    else:
                        self.logger.warning(f"📭 Datos vacíos para {symbol} en intento {attempt + 1}")

            except asyncio.TimeoutError:
                self.stats['timeout_errors'] += 1
                self.logger.warning(f"⏰ Timeout en intento {attempt + 1} para {symbol}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Backoff exponencial
                continue

            except Exception as e:
                self.logger.warning(f"⚠️ Intento {attempt + 1}/{self.max_retries} fallido para {symbol}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                continue

        self.logger.error(f"❌ Todos los intentos fallidos para obtener datos de {symbol}")
        return None

    async def _reevaluate_single_signal(self, signal: Dict, current_data: Dict,
                                      market_context: Dict, risk_metrics: Dict) -> ReevaluationResult:
        """
        Re-evalúa una señal individual considerando:
        1. Contexto de mercado actual
        2. Calidad del punto de entrada
        3. Confirmación técnica
        4. Gestión de riesgo
        """
        symbol = signal['symbol']
        signal_type = signal['type']

        # 1. Verificar confirmación técnica
        technical_confirmation = self._check_technical_confirmation(signal, current_data)

        # 2. Evaluar calidad de entrada
        entry_quality = self._evaluate_entry_quality(signal, current_data, market_context)

        # 3. Calcular confianza inicial
        initial_confidence = self._calculate_initial_confidence(
            signal, technical_confirmation, entry_quality
        )

        # 4. Incorporar gestión de riesgo
        final_confidence = self._incorporate_risk_management(initial_confidence, risk_metrics)

        # 5. Tomar decisión
        is_confirmed = final_confidence >= self.min_confidence
        recommendation = self._generate_recommendation(
            is_confirmed, entry_quality, final_confidence, market_context
        )

        return ReevaluationResult(
            signal=signal,
            is_confirmed=is_confirmed,
            confidence=final_confidence,
            entry_quality=entry_quality,
            adjustment_factors={
                'market_context': market_context,
                'technical_confirmation': technical_confirmation,
                'original_score': signal.get('composite_score', 0),
                'adjusted_score': signal.get('adjusted_score', 0),
                'risk_metrics': risk_metrics,
                'signal_age_minutes': self._calculate_signal_age(signal)
            },
            recommendation=recommendation,
            reevaluation_timestamp=datetime.now()
        )

    async def _get_current_market_data(self, symbol: str) -> Optional[Dict]:
        """Obtiene datos de mercado actualizados"""
        try:
            # Obtener datos para timeframes relevantes
            timeframes = ['5m', '15m']  # Timeframes para confirmación
            market_data = {}

            for tf in timeframes:
                data = await self.data_loader.get_ohlcv(symbol, tf, limit=50)
                if data is not None and len(data) > 0:
                    market_data[tf] = data

            return market_data if market_data else None

        except Exception as e:
            self.logger.error(f"❌ Error obteniendo datos para {symbol}: {e}")
            return None

    async def _analyze_market_context(self, symbol: str, current_data: Dict) -> Dict:
        """Analiza el contexto actual del mercado"""
        try:
            context = {
                'volatility': 0.0,
                'trend_strength': 0.0,
                'market_condition': 'NORMAL',
                'volume_analysis': 'NORMAL',
                'timestamp': datetime.now().isoformat()
            }

            # Usar timeframe de 15m para análisis de contexto
            if '15m' in current_data:
                data_15m = current_data['15m']

                # Volatilidad (ATR normalizado)
                atr = self._calculate_atr_quick(data_15m)
                current_price = data_15m['close'].iloc[-1]
                volatility = atr / current_price if current_price > 0 else 0
                context['volatility'] = volatility

                # Fuerza de tendencia
                trend_strength = self._calculate_trend_strength(data_15m)
                context['trend_strength'] = trend_strength

                # Condición de mercado basada en volatilidad
                if volatility > 0.03:
                    context['market_condition'] = 'HIGH_VOLATILITY'
                elif volatility < 0.005:
                    context['market_condition'] = 'LOW_VOLATILITY'

                # Análisis de volumen
                volume_analysis = self._analyze_volume(data_15m)
                context['volume_analysis'] = volume_analysis

                # Información adicional de precio
                context['current_price'] = current_price
                context['price_change_24h'] = self._calculate_24h_change(data_15m)

            return context

        except Exception as e:
            self.logger.error(f"❌ Error analizando contexto de mercado para {symbol}: {e}")
            return {
                'market_condition': 'UNKNOWN',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _check_technical_confirmation(self, signal: Dict, current_data: Dict) -> Dict:
        """Verifica la confirmación técnica actual"""
        confirmation = {
            'timeframe_alignment': 0.0,
            'indicator_confirmation': 0.0,
            'price_action': 0.0,
            'overall_score': 0.0,
            'details': {}
        }

        try:
            signal_type = signal['type']
            original_tf_signals = signal.get('timeframe_signals', {})

            # Verificar alineación de timeframes
            timeframe_match = 0
            total_timeframes = 0
            alignment_details = {}

            for tf, data in current_data.items():
                if tf in original_tf_signals:
                    current_trend = self._get_current_trend(data, tf)
                    original_trend = original_tf_signals[tf].get('indicators', {}).get('trend')

                    is_aligned = current_trend == original_trend
                    if is_aligned:
                        timeframe_match += 1

                    alignment_details[tf] = {
                        'current_trend': current_trend,
                        'original_trend': original_trend,
                        'aligned': is_aligned
                    }
                    total_timeframes += 1

            confirmation['timeframe_alignment'] = timeframe_match / total_timeframes if total_timeframes > 0 else 0
            confirmation['details']['timeframe_alignment'] = alignment_details

            # Confirmación de indicadores
            indicator_confirmation, indicator_details = self._check_indicator_confirmation(signal, current_data)
            confirmation['indicator_confirmation'] = indicator_confirmation
            confirmation['details']['indicators'] = indicator_details

            # Price action
            price_action_score, price_action_details = self._analyze_price_action(signal, current_data)
            confirmation['price_action'] = price_action_score
            confirmation['details']['price_action'] = price_action_details

            # Score general
            overall_score = (
                confirmation['timeframe_alignment'] * 0.4 +
                confirmation['indicator_confirmation'] * 0.4 +
                confirmation['price_action'] * 0.2
            )
            confirmation['overall_score'] = overall_score

            return confirmation

        except Exception as e:
            self.logger.error(f"❌ Error en confirmación técnica: {e}")
            return confirmation

    def _evaluate_entry_quality(self, signal: Dict, current_data: Dict, market_context: Dict) -> str:
        """Evalúa la calidad del punto de entrada actual"""
        try:
            quality_score = 0.0
            signal_type = signal['type']
            evaluation_details = {}

            # 1. Posición relativa en tendencia
            trend_score, trend_details = self._evaluate_trend_position(signal, current_data)
            quality_score += trend_score * 0.3
            evaluation_details['trend_position'] = trend_details

            # 2. Niveles de soporte/resistencia
            sr_score, sr_details = self._evaluate_support_resistance(signal, current_data)
            quality_score += sr_score * 0.3
            evaluation_details['support_resistance'] = sr_details

            # 3. Condiciones de volatilidad
            volatility_score, vol_details = self._evaluate_volatility_conditions(market_context)
            quality_score += volatility_score * 0.2
            evaluation_details['volatility'] = vol_details

            # 4. Momentum actual
            momentum_score, momentum_details = self._evaluate_current_momentum(signal, current_data)
            quality_score += momentum_score * 0.2
            evaluation_details['momentum'] = momentum_details

            # Guardar detalles de evaluación en el log para debugging
            if quality_score > self.quality_thresholds['good']:
                self.logger.debug(f"📊 Evaluación calidad entrada {signal['symbol']}: {evaluation_details}")

            # Determinar calidad basada en score
            if quality_score >= self.quality_thresholds['excellent']:
                return "EXCELLENT"
            elif quality_score >= self.quality_thresholds['good']:
                return "GOOD"
            elif quality_score >= self.quality_thresholds['poor']:
                return "POOR"
            else:
                return "REJECTED"

        except Exception as e:
            self.logger.error(f"❌ Error evaluando calidad de entrada para {signal['symbol']}: {e}")
            return "REJECTED"

    def _calculate_initial_confidence(self, signal: Dict, technical_confirmation: Dict,
                                   entry_quality: str) -> float:
        """Calcula la confianza inicial antes de ajustar por riesgo"""
        try:
            base_confidence = signal.get('composite_score', 0) / 100.0

            # Ajustar por confirmación técnica
            tech_multiplier = technical_confirmation.get('overall_score', 0.5)
            base_confidence *= tech_multiplier

            # Ajustar por calidad de entrada
            quality_multipliers = {
                "EXCELLENT": 1.2,
                "GOOD": 1.0,
                "POOR": 0.7,
                "REJECTED": 0.3
            }
            base_confidence *= quality_multipliers.get(entry_quality, 0.5)

            return min(max(base_confidence, 0.0), 1.0)

        except Exception as e:
            self.logger.error(f"❌ Error calculando confianza inicial: {e}")
            return 0.0

    def _incorporate_risk_management(self, confidence: float, risk_metrics: Dict) -> float:
        """Incorpora métricas de riesgo en la confianza final"""
        try:
            risk_score = risk_metrics.get('risk_score', 0.0)
            volatility_penalty = risk_metrics.get('volatility_penalty', 0.0)
            liquidity_risk = risk_metrics.get('liquidity_risk', 0.0)

            # Calcular factor de ajuste de riesgo
            risk_factor = 1.0 - (risk_score * 0.3 + volatility_penalty * 0.4 + liquidity_risk * 0.3)

            # Aplicar ajuste de riesgo
            adjusted_confidence = confidence * risk_factor

            return min(max(adjusted_confidence, 0.0), 1.0)
        except Exception as e:
            self.logger.error(f"❌ Error incorporando gestión de riesgo: {e}")
            return confidence

    def _generate_recommendation(self, is_confirmed: bool, entry_quality: str,
                               confidence: float, market_context: Dict) -> str:
        """Genera recomendación final"""
        if not is_confirmed:
            return "CANCEL"

        if entry_quality == "REJECTED":
            return "CANCEL"

        # Condiciones de mercado adversas
        market_condition = market_context.get('market_condition', 'NORMAL')
        if market_condition == 'HIGH_VOLATILITY' and confidence < 0.8:
            return "WAIT"

        # Recomendaciones basadas en calidad y confianza
        if entry_quality == "EXCELLENT":
            if confidence >= 0.8:
                return "EXECUTE"
            elif confidence >= 0.7:
                return "WAIT"  # Esperar confirmación adicional
            else:
                return "CANCEL"

        elif entry_quality == "GOOD":
            if confidence >= 0.75:
                return "EXECUTE"
            elif confidence >= 0.65:
                return "WAIT"
            else:
                return "CANCEL"

        elif entry_quality == "POOR":
            if confidence >= 0.8:
                return "WAIT"  # Solo en casos muy excepcionales
            else:
                return "CANCEL"

        else:
            return "CANCEL"

    # ===== MÉTODOS DE ANÁLISIS TÉCNICO =====

    def _calculate_atr_quick(self, data: pd.DataFrame) -> float:
        """Calcula ATR rápidamente"""
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

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calcula fuerza de la tendencia"""
        try:
            if len(data) < 20:
                return 0.5

            prices = data['close'].values
            sma_10 = np.mean(prices[-10:])
            sma_20 = np.mean(prices[-20:])

            # Dirección de la tendencia
            direction = 1 if sma_10 > sma_20 else -1

            # Fuerza basada en la separación
            separation = abs(sma_10 - sma_20) / sma_20 if sma_20 > 0 else 0

            return min(separation * 10 * direction, 1.0)
        except:
            return 0.5

    def _analyze_volume(self, data: pd.DataFrame) -> str:
        """Analiza condiciones de volumen"""
        try:
            if len(data) < 5:
                return "UNKNOWN"

            volumes = data['volume'].values
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[-5:])

            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            if volume_ratio > 1.5:
                return "HIGH_VOLUME"
            elif volume_ratio < 0.7:
                return "LOW_VOLUME"
            else:
                return "NORMAL_VOLUME"
        except:
            return "UNKNOWN"

    def _calculate_24h_change(self, data: pd.DataFrame) -> float:
        """Calcula cambio porcentual en 24 horas"""
        try:
            if len(data) < 24 * 4:  # Aproximadamente 24 horas en velas de 15m
                return 0.0

            current_price = data['close'].iloc[-1]
            price_24h_ago = data['close'].iloc[-24 * 4]  # 96 velas atrás para 15m

            return (current_price - price_24h_ago) / price_24h_ago if price_24h_ago > 0 else 0.0
        except:
            return 0.0

    def _get_current_trend(self, data: pd.DataFrame, timeframe: str) -> str:
        """Obtiene la tendencia actual"""
        try:
            if len(data) < 10:
                return "NEUTRAL"

            prices = data['close'].values
            short_ma = np.mean(prices[-5:])
            long_ma = np.mean(prices[-10:])

            if short_ma > long_ma * 1.002:
                return "BULLISH"
            elif short_ma < long_ma * 0.998:
                return "BEARISH"
            else:
                return "NEUTRAL"
        except:
            return "NEUTRAL"

    def _check_indicator_confirmation(self, signal: Dict, current_data: Dict) -> Tuple[float, Dict]:
        """Verifica confirmación de indicadores"""
        try:
            confirmation_score = 0.0
            signal_type = signal['type']
            details = {}

            for tf, data in current_data.items():
                tf_details = {}

                # RSI confirmation
                rsi = self._calculate_rsi_quick(data)
                rsi_confirm = (signal_type == 'LONG' and rsi < 60) or (signal_type == 'SHORT' and rsi > 40)
                if rsi_confirm:
                    confirmation_score += 0.2
                tf_details['rsi'] = {'value': rsi, 'confirmed': rsi_confirm}

                # Price position confirmation
                current_price = data['close'].iloc[-1]
                sma_20 = data['close'].rolling(20).mean().iloc[-1]
                price_confirm = (signal_type == 'LONG' and current_price > sma_20) or (signal_type == 'SHORT' and current_price < sma_20)
                if price_confirm:
                    confirmation_score += 0.2
                tf_details['price_position'] = {'current': current_price, 'sma_20': sma_20, 'confirmed': price_confirm}

                details[tf] = tf_details

            # Normalizar score
            total_timeframes = len(current_data)
            normalized_score = confirmation_score / total_timeframes if total_timeframes > 0 else 0.0

            return normalized_score, details

        except Exception as e:
            self.logger.error(f"❌ Error en confirmación de indicadores: {e}")
            return 0.5, {'error': str(e)}

    def _analyze_price_action(self, signal: Dict, current_data: Dict) -> Tuple[float, Dict]:
        """Analiza price action para confirmación"""
        try:
            score = 0.0
            signal_type = signal['type']
            details = {}

            for tf, data in current_data.items():
                tf_details = {}
                if len(data) < 3:
                    continue

                # Analizar velas recientes
                recent_candles = data.tail(3)

                # Verificar patrones de continuación
                continuation_pattern = self._check_continuation_pattern(recent_candles, signal_type)
                if continuation_pattern:
                    score += 0.3
                tf_details['continuation_pattern'] = continuation_pattern

                # Verificar momentum
                price_changes = recent_candles['close'].pct_change().dropna()
                positive_changes = sum(1 for change in price_changes if change > 0)
                momentum_good = (signal_type == 'LONG' and positive_changes >= 2) or (signal_type == 'SHORT' and positive_changes <= 1)
                if momentum_good:
                    score += 0.2
                tf_details['momentum'] = {'positive_changes': positive_changes, 'good': momentum_good}

                details[tf] = tf_details

            return min(score, 1.0), details

        except Exception as e:
            self.logger.error(f"❌ Error analizando price action: {e}")
            return 0.5, {'error': str(e)}

    def _evaluate_trend_position(self, signal: Dict, current_data: Dict) -> Tuple[float, Dict]:
        """Evalúa posición relativa en la tendencia"""
        try:
            signal_type = signal['type']
            score = 0.0
            details = {}

            for tf, data in current_data.items():
                tf_details = {}
                if len(data) < 20:
                    continue

                current_price = data['close'].iloc[-1]
                sma_20 = data['close'].rolling(20).mean().iloc[-1]
                sma_50 = data['close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else sma_20

                # Posición respecto a medias móviles
                if signal_type == 'LONG':
                    if current_price > sma_20 > sma_50:
                        score += 0.5  # Tendencia alcista fuerte
                        trend_strength = "STRONG_BULLISH"
                    elif current_price > sma_20:
                        score += 0.3  # Tendencia alcista débil
                        trend_strength = "WEAK_BULLISH"
                    else:
                        trend_strength = "NOT_BULLISH"
                else:  # SHORT
                    if current_price < sma_20 < sma_50:
                        score += 0.5  # Tendencia bajista fuerte
                        trend_strength = "STRONG_BEARISH"
                    elif current_price < sma_20:
                        score += 0.3  # Tendencia bajista débil
                        trend_strength = "WEAK_BEARISH"
                    else:
                        trend_strength = "NOT_BEARISH"

                tf_details = {
                    'current_price': current_price,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'trend_strength': trend_strength
                }
                details[tf] = tf_details

            # Promediar scores de todos los timeframes
            avg_score = score / len(current_data) if current_data else 0.0
            return min(avg_score, 1.0), details

        except Exception as e:
            self.logger.error(f"❌ Error evaluando posición de tendencia: {e}")
            return 0.5, {'error': str(e)}

    def _evaluate_support_resistance(self, signal: Dict, current_data: Dict) -> Tuple[float, Dict]:
        """Evalúa niveles de soporte/resistencia"""
        try:
            signal_type = signal['type']
            score = 0.0
            details = {}

            for tf, data in current_data.items():
                tf_details = {}
                if len(data) < 30:
                    continue

                current_price = data['close'].iloc[-1]
                highs = data['high'].tail(20).values
                lows = data['low'].tail(20).values

                # Identificar niveles clave
                resistance_levels = []
                support_levels = []

                # Simple resistance detection (highs)
                for high in highs:
                    if high > current_price * 1.01:  # 1% por encima
                        resistance_levels.append(high)

                # Simple support detection (lows)
                for low in lows:
                    if low < current_price * 0.99:  # 1% por debajo
                        support_levels.append(low)

                # Evaluar proximidad a niveles clave
                if signal_type == 'LONG':
                    # Buscar soporte cerca
                    if support_levels:
                        closest_support = min(support_levels, key=lambda x: abs(x - current_price))
                        distance = abs(closest_support - current_price) / current_price

                        if distance < 0.005:  # Menos de 0.5%
                            score += 0.7  # Buen punto de entrada
                            support_quality = "EXCELLENT"
                        elif distance < 0.01:  # Menos de 1%
                            score += 0.4  # Aceptable
                            support_quality = "GOOD"
                        else:
                            support_quality = "POOR"
                    else:
                        support_quality = "NO_SUPPORT"
                else:  # SHORT
                    # Buscar resistencia cerca
                    if resistance_levels:
                        closest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
                        distance = abs(closest_resistance - current_price) / current_price

                        if distance < 0.005:  # Menos de 0.5%
                            score += 0.7  # Buen punto de entrada
                            resistance_quality = "EXCELLENT"
                        elif distance < 0.01:  # Menos de 1%
                            score += 0.4  # Aceptable
                            resistance_quality = "GOOD"
                        else:
                            resistance_quality = "POOR"
                    else:
                        resistance_quality = "NO_RESISTANCE"

                tf_details = {
                    'support_levels_count': len(support_levels),
                    'resistance_levels_count': len(resistance_levels),
                    'quality': support_quality if signal_type == 'LONG' else resistance_quality
                }
                details[tf] = tf_details

            avg_score = score / len(current_data) if current_data else 0.0
            return min(avg_score, 1.0), details

        except Exception as e:
            self.logger.error(f"❌ Error evaluando soporte/resistencia: {e}")
            return 0.5, {'error': str(e)}

    def _evaluate_volatility_conditions(self, market_context: Dict) -> Tuple[float, Dict]:
        """Evalúa condiciones de volatilidad"""
        try:
            volatility = market_context.get('volatility', 0)
            details = {'volatility': volatility}

            if 0.005 <= volatility <= 0.02:  # Volatilidad ideal
                return 1.0, {**details, 'condition': 'IDEAL'}
            elif 0.002 <= volatility < 0.005:  # Baja volatilidad
                return 0.7, {**details, 'condition': 'LOW'}
            elif 0.02 < volatility <= 0.04:  # Alta volatilidad pero manejable
                return 0.8, {**details, 'condition': 'HIGH_MANAGEABLE'}
            else:  # Volatilidad extrema
                return 0.3, {**details, 'condition': 'EXTREME'}
        except Exception as e:
            self.logger.error(f"❌ Error evaluando condiciones de volatilidad: {e}")
            return 0.5, {'error': str(e)}

    def _evaluate_current_momentum(self, signal: Dict, current_data: Dict) -> Tuple[float, Dict]:
        """Evalúa momentum actual"""
        try:
            signal_type = signal['type']
            score = 0.0
            details = {}

            for tf, data in current_data.items():
                tf_details = {}
                if len(data) < 10:
                    continue

                # Calcular momentum
                momentum = data['close'].pct_change().tail(5).mean()
                momentum_good = (signal_type == 'LONG' and momentum > 0) or (signal_type == 'SHORT' and momentum < 0)

                if momentum_good:
                    score += 0.5

                tf_details = {
                    'momentum': momentum,
                    'good': momentum_good
                }
                details[tf] = tf_details

            avg_score = score / len(current_data) if current_data else 0.0
            return min(avg_score, 1.0), details

        except Exception as e:
            self.logger.error(f"❌ Error evaluando momentum actual: {e}")
            return 0.5, {'error': str(e)}

    def _evaluate_risk_metrics(self, signal: Dict, current_data: Dict) -> Dict:
        """Evalúa métricas de riesgo adicionales"""
        try:
            risk_metrics = {
                'risk_score': 0.0,
                'max_drawdown': 0.0,
                'volatility_penalty': 0.0,
                'liquidity_risk': 0.0,
                'details': {}
            }

            symbol = signal['symbol']
            signal_type = signal['type']

            # Evaluar riesgo basado en drawdown potencial
            for tf, data in current_data.items():
                if len(data) < 50:
                    continue

                # Calcular drawdown potencial
                current_price = data['close'].iloc[-1]
                rolling_max = data['close'].expanding().max()
                drawdown = (current_price - rolling_max) / rolling_max
                max_drawdown = abs(drawdown.min())

                risk_metrics['max_drawdown'] = max(risk_metrics['max_drawdown'], max_drawdown)

                # Penalizar por drawdown alto
                if max_drawdown > 0.1:  # 10%
                    risk_metrics['risk_score'] += 0.3
                elif max_drawdown > 0.05:  # 5%
                    risk_metrics['risk_score'] += 0.2

            # Evaluar riesgo de volatilidad
            volatility = self._calculate_atr_quick(current_data.get('15m', pd.DataFrame()))
            risk_metrics['details']['volatility'] = volatility
            if volatility > 0.03:  # Alta volatilidad
                risk_metrics['volatility_penalty'] = 0.2
            elif volatility > 0.05:  # Volatilidad muy alta
                risk_metrics['volatility_penalty'] = 0.4

            # Evaluar riesgo de liquidez
            volume_data = current_data.get('15m', pd.DataFrame())
            if not volume_data.empty and 'volume' in volume_data.columns:
                volume = volume_data['volume'].iloc[-1]
                risk_metrics['details']['volume'] = volume
                if volume < 100000:  # Bajo volumen
                    risk_metrics['liquidity_risk'] = 0.3
                elif volume < 50000:  # Muy bajo volumen
                    risk_metrics['liquidity_risk'] = 0.5

            # Normalizar risk_score
            risk_metrics['risk_score'] = min(risk_metrics['risk_score'], 1.0)

            return risk_metrics

        except Exception as e:
            self.logger.error(f"❌ Error evaluando métricas de riesgo: {e}")
            return {'risk_score': 0.5, 'error': str(e)}

    def _check_continuation_pattern(self, candles: pd.DataFrame, signal_type: str) -> bool:
        """Verifica patrones de continuación"""
        try:
            if len(candles) < 3:
                return False

            # Verificar patrón de tres velas
            closes = candles['close'].values
            highs = candles['high'].values
            lows = candles['low'].values

            if signal_type == 'LONG':
                # Patrón alcista: velas verdes con mínimos más altos
                return (closes[0] < closes[1] < closes[2] and
                        lows[0] < lows[1] < lows[2])
            else:
                # Patrón bajista: velas rojas con máximos más bajos
                return (closes[0] > closes[1] > closes[2] and
                        highs[0] > highs[1] > highs[2])
        except:
            return False

    def _calculate_rsi_quick(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calcula RSI rápidamente"""
        try:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50
        except:
            return 50

    def _calculate_signal_age(self, signal: Dict) -> float:
        """Calcula la edad de la señal en minutos"""
        try:
            signal_time = datetime.fromisoformat(signal.get('timestamp', datetime.now().isoformat()))
            return (datetime.now() - signal_time).total_seconds() / 60
        except:
            return 0.0

    # ===== CIRCUIT BREAKER Y MANEJO DE ERRORES =====

    def _is_circuit_open(self) -> bool:
        """Verifica si el circuit breaker está abierto"""
        now = datetime.now()
        # Limpiar errores antiguos (último minuto)
        self.error_timestamps = [ts for ts in self.error_timestamps
                               if (now - ts).total_seconds() < 60]

        is_open = len(self.error_timestamps) >= self.max_errors_per_minute
        if is_open:
            self.logger.warning(f"🔒 Circuit breaker ABIERTO - Errores recientes: {len(self.error_timestamps)}")

        return is_open

    def _record_error(self):
        """Registra un error para el circuit breaker"""
        self.error_timestamps.append(datetime.now())
        # Mantener solo los últimos N errores para evitar crecimiento infinito
        if len(self.error_timestamps) > self.max_errors_per_minute * 2:
            self.error_timestamps = self.error_timestamps[-self.max_errors_per_minute:]

    def _clean_old_cache(self):
        """Limpia cache antiguo"""
        now = datetime.now()
        keys_to_remove = []
        for key, (data, timestamp) in self.market_data_cache.items():
            if now - timestamp > self.cache_ttl:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.market_data_cache[key]

        if keys_to_remove:
            self.logger.debug(f"🧹 Cache limpiado: {len(keys_to_remove)} entradas removidas")

    def _create_rejected_result(self, signal: Dict, reason: str) -> ReevaluationResult:
        """Crea un resultado de rechazo"""
        return ReevaluationResult(
            signal=signal,
            is_confirmed=False,
            confidence=0.0,
            entry_quality="REJECTED",
            adjustment_factors={'rejection_reason': reason},
            recommendation="CANCEL",
            reevaluation_timestamp=datetime.now()
        )

    def _log_reevaluation_summary(self, results: List[ReevaluationResult]):
        """Log detallado para monitoreo en producción"""
        if not results:
            return

        confirmed = [r for r in results if r.is_confirmed]
        rejected = [r for r in results if not r.is_confirmed]
        execute_recommendations = [r for r in confirmed if r.recommendation == "EXECUTE"]
        excellent_quality = [r for r in confirmed if r.entry_quality == "EXCELLENT"]

        confirmation_rate = len(confirmed) / len(results) * 100
        execute_rate = len(execute_recommendations) / len(confirmed) * 100 if confirmed else 0

        self.logger.info(
            f"📊 RESUMEN RE-EVALUACIÓN - "
            f"Confirmadas: {len(confirmed)}/{len(results)} "
            f"({confirmation_rate:.1f}%) - "
            f"Excelentes: {len(excellent_quality)} - "
            f"EXECUTE: {len(execute_recommendations)} "
            f"({execute_rate:.1f}% de confirmadas)"
        )

        # Log detallado de señales confirmadas
        for result in confirmed:
            self.logger.info(
                f"🎯 CONFIRMADA: {result.signal['symbol']} {result.signal['type']} - "
                f"Calidad: {result.entry_quality} - Confianza: {result.confidence:.2f} - "
                f"Recomendación: {result.recommendation}"
            )

    def get_stats(self) -> Dict:
        """Obtiene estadísticas detalladas del re-evaluador"""
        total_reevaluations = max(self.stats['total_reevaluations'], 1)
        return {
            **self.stats,
            'confirmation_rate': self.stats['signals_confirmed'] / total_reevaluations,
            'error_rate': self.stats['error_count'] / total_reevaluations,
            'circuit_breaker_active': self._is_circuit_open(),
            'recent_errors': len(self.error_timestamps),
            'cache_size': len(self.market_data_cache),
            'timestamp': datetime.now().isoformat()
        }

    def reset_stats(self):
        """Reinicia las estadísticas (útil para tests)"""
        self.stats = {
            'total_reevaluations': 0,
            'signals_confirmed': 0,
            'signals_rejected': 0,
            'last_reevaluation': None,
            'retry_count': 0,
            'error_count': 0,
            'circuit_breaker_triggers': 0,
            'timeout_errors': 0
        }
        self.error_timestamps = []
        self.market_data_cache = {}

    def health_check(self) -> Dict:
        """Verifica la salud del re-evaluador"""
        stats = self.get_stats()
        return {
            'status': 'HEALTHY' if stats['error_rate'] < 0.1 and not self._is_circuit_open() else 'DEGRADED',
            'stats': stats,
            'circuit_breaker': 'OPEN' if self._is_circuit_open() else 'CLOSED',
            'last_activity': self.stats['last_reevaluation'],
            'cache_health': f"{len(self.market_data_cache)} entradas"
        }


# Función de utilidad para crear instancia preconfigurada
def create_production_reevaluator(config: Dict, data_loader) -> OptimizedSignalReevaluator:
    """Crea una instancia del re-evaluador configurada para producción"""
    return OptimizedSignalReevaluator(
        config=config,
        data_loader=data_loader,
        max_retries=config.get('signal_reevaluator', {}).get('performance', {}).get('max_retries', 3),
        retry_delay=config.get('signal_reevaluator', {}).get('performance', {}).get('retry_delay', 1.0)
    )