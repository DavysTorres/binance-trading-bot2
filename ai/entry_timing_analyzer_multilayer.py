

# ai/entry_timing_analyzer_multilayer.py
# ===============================================================
# 🎯 Entry Timing Analyzer (versión mejorada: 3m + 5m + 15m)
# ---------------------------------------------------------------
# Incluye mejoras basadas en análisis de pruebas:
#   ✅ Sistema de scoring ajustado y configurable
#   ✅ Penalización fuerte de inconsistencias técnicas
#   ✅ Detección mejorada de etapas específicas
#   ✅ Manejo robusto de estado y contexto
#   ✅ Detección avanzada de patrones
#   ✅ Validación de datos mejorada
#   ✅ Optimización con memoización
#   ✅ Manejo detallado de errores y logging
# ===============================================================

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('timing_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===============================================================
# 🔹 Indicadores auxiliares (optimizados con memoización)
# ===============================================================

@lru_cache(maxsize=128)
def ema(series_hash: Tuple[float, ...], n: int) -> np.ndarray:
    """Calcula EMA con memoización para evitar recálculos"""
    try:
        series = pd.Series(series_hash)
        return series.ewm(span=n, adjust=False).mean().values
    except Exception as e:
        logger.error(f"Error en EMA: {e}")
        return np.array([])

@lru_cache(maxsize=128)
def rsi(series_hash: Tuple[float, ...], n: int = 14) -> np.ndarray:
    """Calcula RSI con memoización"""
    try:
        series = pd.Series(series_hash)
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        rs = (up.rolling(n).mean()) / (down.rolling(n).mean())
        return (100 - (100 / (1 + rs))).fillna(50).values
    except Exception as e:
        logger.error(f"Error en RSI: {e}")
        return np.array([])

@lru_cache(maxsize=128)
def macd(series_hash: Tuple[float, ...], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula MACD con memoización"""
    try:
        series = pd.Series(series_hash)
        ema_fast = ema(tuple(series.values), fast)
        ema_slow = ema(tuple(series.values), slow)
        line = ema_fast - ema_slow
        sig = ema(tuple(line), signal)
        hist = line - sig
        return line, sig, hist
    except Exception as e:
        logger.error(f"Error en MACD: {e}")
        return np.array([]), np.array([]), np.array([])

@lru_cache(maxsize=128)
def bollinger(series_hash: Tuple[float, ...], n: int = 20, k: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula Bandas de Bollinger con memoización"""
    try:
        series = pd.Series(series_hash)
        ma = series.rolling(n).mean().values
        sd = series.rolling(n).std().values
        upper = ma + k * sd
        lower = ma - k * sd
        return upper, ma, lower
    except Exception as e:
        logger.error(f"Error en Bandas de Bollinger: {e}")
        return np.array([]), np.array([]), np.array([])

# ===============================================================
# 🔍 Validación robusta de dataframes
# ===============================================================

def validate_dataframe(df: pd.DataFrame, min_length: int = 30, required_cols: List[str] = None) -> bool:
    """Valida dataframe con chequeos completos"""
    if required_cols is None:
        required_cols = ['open', 'high', 'low', 'close']
    
    try:
        if df is None or len(df) < min_length:
            logger.warning(f"DataFrame inválido: longitud {len(df) if df else 0} < {min_length}")
            return False
        
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Columnas faltantes: {[col for col in required_cols if col not in df.columns]}")
            return False
        
        if df.isnull().any().any():
            logger.warning("DataFrame contiene valores nulos")
            return False
        
        # Verificar que no haya valores cero en precios
        if (df[['open', 'high', 'low', 'close']] == 0).any().any():
            logger.warning("DataFrame contiene precios cero")
            return False
        
        # Verificar consistencia OHLC
        if (df['high'] < df['low']).any():
            logger.warning("Valores inconsistentes: high < low")
            return False
        
        if (df['high'] < df[['open', 'close']].min(axis=1)).any():
            logger.warning("Valores inconsistentes: high menor que open/close")
            return False
        
        if (df['low'] > df[['open', 'close']].max(axis=1)).any():
            logger.warning("Valores inconsistentes: low mayor que open/close")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error en validación de dataframe: {e}")
        return False

# ===============================================================
# 🧩 Funciones complementarias (mejoradas para reversas y etapas)
# ===============================================================

def _etapa_por_tf_mejorado(df: pd.DataFrame, tendencia: str, rsi_3m: float, macd_hist_3m: float, macd_hist_3m_prev: float) -> str:
    """Clasifica la etapa mejorada con detección específica de pullbacks y rebotes"""

    if not validate_dataframe(df, min_length=20):
        return "indefinido"

    try:
        c = df["close"]
        ema20 = c.rolling(20).mean()

        # CORRECCIÓN: Usar arrays de NumPy correctamente
        ub, mb, lb = bollinger(tuple(c.values), 20, 2)
        dist = (c.iloc[-1] - ema20.iloc[-1]) / max(ema20.iloc[-1], 1e-9)

        # conteo de velas consecutivas
        tail = df.tail(4)
        up_cnt = (tail["close"] > tail["open"]).sum()
        dn_cnt = (tail["close"] < tail["open"]).sum()

        # Detección específica de etapas
        if tendencia == "bullish":
            # Rebote técnico en sobreventa
            if rsi_3m < 30 and macd_hist_3m > macd_hist_3m_prev:
                return "rebote"

            # Pullback saludable (RSI < 50 y tendencia alcista)
            if rsi_3m < 50 and dist < -0.01:
                return "pullback"

            # Sobreextensión
            if rsi_3m > 70 or dist > 0.02 or up_cnt >= 4:
                return "extendido"
        else:  # bearish
            # Rebote técnico en sobrecompra
            if rsi_3m > 70 and macd_hist_3m < macd_hist_3m_prev:
                return "rebote"

            # Pullback saludable (RSI > 50 y tendencia bajista)
            if rsi_3m > 50 and dist > 0.01:
                return "pullback"

            # Sobreextensión
            if rsi_3m < 30 or dist < -0.02 or dn_cnt >= 4:
                return "extendido"

        # CORRECCIÓN: Usar indexación estándar para arrays de NumPy
        if c.iloc[-1] > ub[-1] or dist > 0.02 or up_cnt >= 4:
            return "final" if c.iloc[-1] > ub[-1] else "extendido"
        if c.iloc[-1] < lb[-1] or dist < -0.02 or dn_cnt >= 4:
            return "final" if c.iloc[-1] < lb[-1] else "extendido"

        return "inicio"
    except Exception as e:
        logger.error(f"Error al determinar etapa mejorada: {e}")
        return "error"

def _reversa_ult2(df: pd.DataFrame) -> str:
    """Devuelve 'bullish'|'bearish'|'none' según patrones en las últimas 2 velas."""

    if not validate_dataframe(df, min_length=2):
        return "none"

    try:
        seg = df.tail(2)
        # engulfings
        bull_eng = (seg["close"].iloc[-2] < seg["open"].iloc[-2] and
                    seg["close"].iloc[-1] > seg["open"].iloc[-1] and
                    seg["open"].iloc[-1] <= seg["close"].iloc[-2] and
                    seg["close"].iloc[-1] >= seg["open"].iloc[-2])
        bear_eng = (seg["close"].iloc[-2] > seg["open"].iloc[-2] and
                    seg["close"].iloc[-1] < seg["open"].iloc[-1] and
                    seg["open"].iloc[-1] >= seg["close"].iloc[-2] and
                    seg["close"].iloc[-1] <= seg["open"].iloc[-2])

        # vela actual tipo hammer/shooting
        o,c,h,l = seg["open"].iloc[-1], seg["close"].iloc[-1], seg["high"].iloc[-1], seg["low"].iloc[-1]
        body = abs(c-o); lower = min(o,c)-l; upper = h-max(o,c)
        hammer = body>0 and lower >= 2*body and upper <= body
        shooting = body>0 and upper >= 2*body and lower <= body

        if bull_eng or hammer:
            return "bullish"
        if bear_eng or shooting:
            return "bearish"
        return "none"
    except Exception as e:
        logger.error(f"Error al detectar reversa: {e}")
        return "none"

def _detect_patrones_reversa(df: pd.DataFrame) -> Dict[str, bool]:
    """Detecta múltiples patrones de reversión con confirmación"""
    if not validate_dataframe(df, min_length=3):
        return {}
    
    try:
        patrones = {}
        seg = df.tail(3)
        
        # Morning Star (reversión bajista a alcista)
        patrones['morning_star'] = (
            seg["close"].iloc[0] < seg["open"].iloc[0] and  # Vela roja
            seg["close"].iloc[1] > seg["open"].iloc[1] and  # Vela verde (gap)
            seg["close"].iloc[2] > seg["close"].iloc[1] and  # Confirmación alcista
            seg["close"].iloc[1] < min(seg["open"].iloc[0], seg["close"].iloc[0])  # Gap bajista
        )
        
        # Evening Star (reversión alcista a bajista)
        patrones['evening_star'] = (
            seg["close"].iloc[0] > seg["open"].iloc[0] and  # Vela verde
            seg["close"].iloc[1] < seg["open"].iloc[1] and  # Vela roja (gap)
            seg["close"].iloc[2] < seg["close"].iloc[1] and  # Confirmación bajista
            seg["close"].iloc[1] > max(seg["open"].iloc[0], seg["close"].iloc[0])  # Gap alcista
        )
        
        # Doji en vela central
        o2, c2 = seg["open"].iloc[1], seg["close"].iloc[1]
        body_size = abs(c2 - o2)
        range_size = seg["high"].iloc[1] - seg["low"].iloc[1]
        patrones['doji'] = body_size < 0.1 * range_size
        
        return patrones
    except Exception as e:
        logger.error(f"Error al detectar patrones de reversión: {e}")
        return {}

# ===============================================================
# 🧠 Sistema de scoring configurable (mejorado)
# ===============================================================

class ScoringSystem:
    """Sistema de scoring configurable para evaluación de timing"""

    def __init__(self):
        # Ponderaciones ajustadas basadas en análisis de pruebas
        self.weights = {
            'total_alignment': 0.25,      # Reducido de 0.3
            'micro_confirmation': 0.08,   # Reducido de 0.1
            'healthy_pullback': 0.15,
            'transition': 0.05,
            'contratendencia_penalty': 0.5,  # Aumentado de 0.4
            'rsi_filter': 0.12,           # Aumentado de 0.1
            'macd_filter': 0.25,          # Aumentado significativamente
            'technical_bounce': 0.15,      # Reducido de 0.2
            'volume_surge': 0.1,          # Nuevo: volumen inusual
            'support_resistance': 0.1,     # Nuevo: niveles clave
            'macd_trend_consistency': 0.3  # Nuevo: penalización por inconsistencia
        }

        # Umbrales ajustados para ser más conservadores
        self.thresholds = {
            'min_score': 0.85,            # Aumentado de 0.75
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_neutral_min': 40,         # Más estricto
            'rsi_neutral_max': 60,         # Más estricto
            'macd_threshold': 0.001,       # Umbral para MACD válido
            'volume_multiplier': 2.0,      # Multiplicador para volumen inusual
            'consistency_threshold': 0.1   # Umbral para consistencia tendencia-MACD
        }

    def calculate_score(self, metrics: Dict) -> float:
        """Calcula score basado en métricas y ponderaciones"""
        try:
            score = 1.0

            # Aplicar ponderaciones según configuración
            if metrics.get('total_alignment', False):
                score += self.weights['total_alignment']
                
            if metrics.get('micro_confirmation', False):
                score += self.weights['micro_confirmation']
                
            if metrics.get('healthy_pullback', False):
                score += self.weights['healthy_pullback']
                
            if metrics.get('transition', False):
                score += self.weights['transition']
                
            if metrics.get('contratendencia', False):
                score -= self.weights['contratendencia_penalty']
                
            if not metrics.get('rsi_ok', True):
                score -= self.weights['rsi_filter']
                
            if not metrics.get('macd_ok', True):
                score -= self.weights['macd_filter']
                
            # Penalización fuerte por inconsistencia entre tendencia y MACD
            if metrics.get('macd_trend_inconsistency', False):
                score -= self.weights['macd_trend_consistency']
                
            if metrics.get('technical_bounce', False):
                score += self.weights['technical_bounce']
                
            if metrics.get('volume_surge', False):
                score += self.weights['volume_surge']
                
            if metrics.get('support_resistance', False):
                score += self.weights['support_resistance']
            
            # Normalizar score entre 0 y 1
            return max(0.0, min(score, 1.0))
        except Exception as e:
            logger.error(f"Error al calcular score: {e}")
            return 0.0

# ===============================================================
# 🧠 Evaluador multilayer con estado y contexto
# ===============================================================

class TimingAnalyzer:
    """Evaluador de timing con manejo de estado y contexto"""
    
    def __init__(self):
        self.scoring_system = ScoringSystem()
        self.state = {
            'last_score': 0.0,
            'trend_evolution': [],
            'signal_history': [],
            'market_regime': None,
            'last_evaluation': None,
            'consecutive_invalid': 0
        }
    
    def update_state(self, evaluation_result: Dict):
        """Actualiza estado con nueva evaluación"""
        try:
            self.state['last_score'] = evaluation_result['score']
            self.state['signal_history'].append(evaluation_result)
            self.state['last_evaluation'] = evaluation_result
            
            # Detectar cambio de régimen de mercado
            if self._detect_regime_change(evaluation_result):
                self.state['market_regime'] = self._determine_regime(evaluation_result)
            
            # Actualizar contador de inválidos consecutivos
            if not evaluation_result.get('valido', False):
                self.state['consecutive_invalid'] += 1
            else:
                self.state['consecutive_invalid'] = 0
                
        except Exception as e:
            logger.error(f"Error al actualizar estado: {e}")
    
    def _detect_regime_change(self, evaluation: Dict) -> bool:
        """Detecta cambio de régimen de mercado"""
        try:
            if len(self.state['signal_history']) < 3:
                return False
                
            # Comparar con evaluaciones anteriores
            recent_scores = [s['score'] for s in self.state['signal_history'][-3:]]
            return max(recent_scores) - min(recent_scores) > 0.3
        except Exception as e:
            logger.error(f"Error al detectar cambio de régimen: {e}")
            return False
    
    def _determine_regime(self, evaluation: Dict) -> str:
        """Determina régimen de mercado actual"""
        try:
            if evaluation['score'] > 0.8:
                return "bullish_strong"
            elif evaluation['score'] < 0.3:
                return "bearish_strong"
            elif evaluation['etapa'] == "transicion":
                return "transition"
            else:
                return "neutral"
        except Exception as e:
            logger.error(f"Error al determinar régimen: {e}")
            return "unknown"
    
    def get_context(self) -> Dict:
        """Devuelve contexto actual del analizador"""
        return self.state
    
    def should_skip_evaluation(self) -> bool:
        """Determina si se debe saltar la evaluación actual"""
        # Si hay demasiadas evaluaciones inválidas consecutivas, pausar
        if self.state['consecutive_invalid'] >= 5:
            logger.warning("Demasiados inválidos consecutivos, pausando evaluaciones")
            return True
        
        # Si el mercado está en régimen de transición, ser más conservador
        if self.state['market_regime'] == "transition":
            logger.info("Régimen de transición detectado, siendo más conservador")
            return False
        
        return False

# ===============================================================
# 🧠 Evaluador principal mejorado
# ===============================================================

def evaluar_timing_multilayer(df_3m: pd.DataFrame, df_5m: pd.DataFrame, df_15m: pd.DataFrame, 
                             direction: str) -> Dict:
    """
    Evalúa el timing en 3m con confirmaciones 5m y 15m.
    Retorna dict detallado con métricas de evaluación.
    """
    # Inicializar analizador con estado
    analyzer = TimingAnalyzer()
    
    try:
        # Validación robusta de datos
        if not validate_dataframe(df_3m):
            return {"valido": False, "razon": "Datos 3m inválidos", "score": 0.0}
        if not validate_dataframe(df_5m):
            return {"valido": False, "razon": "Datos 5m inválidos", "score": 0.0}
        if not validate_dataframe(df_15m):
            return {"valido": False, "razon": "Datos 15m inválidos", "score": 0.0}
        
        # Verificar si se debe saltar la evaluación
        if analyzer.should_skip_evaluation():
            return {"valido": False, "razon": "Evaluación pausada por régimen de mercado", "score": 0.0}

        # ==========================================================
        # Indicadores 3m
        # ==========================================================
        close_3m = df_3m["close"]
        rsi_3m = rsi(tuple(close_3m.values))
        macd_line_3m, macd_sig_3m, macd_hist_3m = macd(tuple(close_3m.values))
        ema_fast_3m = ema(tuple(close_3m.values), 12)
        ema_slow_3m = ema(tuple(close_3m.values), 26)
        tendencia_3m = "bullish" if ema_fast_3m[-1] > ema_slow_3m[-1] else "bearish"

        # Obtener MACD anterior para detección de reversión
        macd_hist_3m_prev = macd_hist_3m[-3] if len(macd_hist_3m) > 3 else macd_hist_3m[-1]

        # ==========================================================
        # Indicadores 5m
        # ==========================================================
        close_5m = df_5m["close"]
        rsi_5m = rsi(tuple(close_5m.values))
        ema_fast_5m = ema(tuple(close_5m.values), 12)
        ema_slow_5m = ema(tuple(close_5m.values), 26)
        macd_line_5m, macd_sig_5m, macd_hist_5m = macd(tuple(close_5m.values))
        tendencia_5m = "bullish" if ema_fast_5m[-1] > ema_slow_5m[-1] else "bearish"

        # ==========================================================
        # Indicadores 15m (predictivo)
        # ==========================================================
        close_15m = df_15m["close"]
        rsi_15m = rsi(tuple(close_15m.values))
        ema_fast_15m = ema(tuple(close_15m.values), 12)
        ema_slow_15m = ema(tuple(close_15m.values), 26)
        tendencia_15m = "bullish" if ema_fast_15m[-1] > ema_slow_15m[-1] else "bearish"

        # ==========================================================
        # 🧭 Inicialización base + Filtro de dirección predominante
        # ==========================================================
        score = 1.0
        etapa = "inicio"
        razon = "Sincronía entre marcos"
        modo_rebote = False
        pullback_saludable = False
        
        # Métricas para scoring
        metrics = {
            'total_alignment': False,
            'micro_confirmation': False,
            'healthy_pullback': False,
            'transition': False,
            'contratendencia': False,
            'rsi_ok': True,
            'macd_ok': True,
            'technical_bounce': False,
            'volume_surge': False,
            'support_resistance': False,
            'macd_trend_inconsistency': False
        }

        predominante = tendencia_15m  # tendencia macro
        direccion_opuesta = (
            (direction == "long" and predominante == "bearish") or
            (direction == "short" and predominante == "bullish")
        )

        if direccion_opuesta:
            score -= 0.4
            etapa = "contratendencia"
            razon = f"Tendencia mayor {predominante} — operación {direction.upper()} en contratendencia"
            metrics['contratendencia'] = True

        # Caso 1: Alineación total
        if tendencia_3m == tendencia_5m == tendencia_15m:
            score += 0.25
            etapa = "inicio"
            metrics['total_alignment'] = True

        # Caso 2: Microconfirmación 3m + 5m
        elif tendencia_3m == tendencia_5m and tendencia_15m != tendencia_3m:
            score += 0.08
            etapa = "temprano"
            razon = "3m+5m alineados — esperando validación 15m"
            metrics['micro_confirmation'] = True

        # Caso 3: Pullback saludable (3m opuesto, 5m y 15m coinciden)
        elif tendencia_3m != tendencia_5m and tendencia_5m == tendencia_15m:
            pullback_saludable = True
            score += 0.15
            etapa = "pullback"
            razon = "Pullback 3m dentro de tendencia mayor"
            metrics['healthy_pullback'] = True

        # Caso 4: Transición
        elif (70 > rsi_3m[-1] > 55 and tendencia_3m != tendencia_5m and tendencia_15m != tendencia_5m) or \
             (45 > rsi_3m[-1] > 20 and tendencia_3m != tendencia_5m and tendencia_15m != tendencia_5m):
            etapa = "transicion"
            score += 0.05
            razon = "Zona mixta — posible cambio de dirección"
            metrics['transition'] = True

        # Caso 5: Contratendencia total
        else:
            score -= 0.5
            etapa = "contratendencia"
            razon = "Marcos no sincronizados"
            metrics['contratendencia'] = True

        # RSI confirmación
        rsi_ok = (analyzer.scoring_system.thresholds['rsi_neutral_min'] < rsi_3m[-1] < 
                  analyzer.scoring_system.thresholds['rsi_neutral_max'])
        if not rsi_ok:
            score -= 0.12
            etapa = "extendido"
            razon = "RSI fuera de zona neutral"
        metrics['rsi_ok'] = rsi_ok

        # MACD coherencia direccional
        macd_ok = (
            (direction == "long" and macd_hist_3m[-1] > analyzer.scoring_system.thresholds['macd_threshold'] and 
             macd_hist_5m[-1] > analyzer.scoring_system.thresholds['macd_threshold']) or
            (direction == "short" and macd_hist_3m[-1] < -analyzer.scoring_system.thresholds['macd_threshold'] and 
             macd_hist_5m[-1] < -analyzer.scoring_system.thresholds['macd_threshold'])
        )
        if not macd_ok:
            score -= 0.25
            razon = "MACD no confirma dirección"
        metrics['macd_ok'] = macd_ok

        # 🔥 CORRECCIÓN: Penalización fuerte por inconsistencia entre tendencia y MACD
        macd_trend_inconsistency = False
        if (tendencia_3m == "bullish" and macd_hist_3m[-1] < -analyzer.scoring_system.thresholds['consistency_threshold']) or \
           (tendencia_3m == "bearish" and macd_hist_3m[-1] > analyzer.scoring_system.thresholds['consistency_threshold']):
            macd_trend_inconsistency = True
            score -= 0.3
            razon += " - Inconsistencia entre tendencia y MACD"
        metrics['macd_trend_inconsistency'] = macd_trend_inconsistency

        # Detectar rebote técnico
        if direction == "long" and rsi_3m[-1] < analyzer.scoring_system.thresholds['rsi_oversold'] and \
           macd_hist_3m[-1] > macd_hist_3m_prev:
            modo_rebote = True
            etapa = "rebote"
            score += 0.15
            razon = "Rebote técnico anticipado"
            metrics['technical_bounce'] = True
        elif direction == "short" and rsi_3m[-1] > analyzer.scoring_system.thresholds['rsi_overbought'] and \
             macd_hist_3m[-1] < macd_hist_3m_prev:
            modo_rebote = True
            etapa = "rebote"
            score += 0.15
            razon = "Rebounce técnico anticipado"
            metrics['technical_bounce'] = True

        # Detectar volumen inusual
        if len(df_3m) > 20:
            avg_volume = df_3m['volume'].tail(20).mean()
            current_volume = df_3m['volume'].iloc[-1]
            if current_volume > analyzer.scoring_system.thresholds['volume_multiplier'] * avg_volume:
                score += 0.1
                metrics['volume_surge'] = True

        # Detectar niveles de soporte/resistencia
        if len(df_3m) > 50:
            recent_high = df_3m['high'].tail(20).max()
            recent_low = df_3m['low'].tail(20).min()
            current_price = df_3m['close'].iloc[-1]
            
            if abs(current_price - recent_high) / recent_high < 0.01:
                score += 0.1
                metrics['support_resistance'] = True
            elif abs(current_price - recent_low) / recent_low < 0.01:
                score += 0.1
                metrics['support_resistance'] = True

        # ==========================================================
        # 🔄 Etapas mejoradas con detección específica
        # ==========================================================
        etapa_3m = _etapa_por_tf_mejorado(df_3m, tendencia_3m, rsi_3m[-1], macd_hist_3m[-1], macd_hist_3m_prev)
        etapa_5m = _etapa_por_tf_mejorado(df_5m, tendencia_5m, rsi_5m[-1], macd_hist_5m[-1], 
                                         macd_hist_5m[-3] if len(macd_hist_5m) > 3 else macd_hist_5m[-1])
        
        # Actualizar etapa principal según detección específica
        if etapa_3m in ["pullback", "rebote"]:
            etapa = etapa_3m
        elif etapa_5m in ["pullback", "rebote"]:
            etapa = etapa_5m

        reversa_ult2 = _reversa_ult2(df_3m)
        
        # Detectar patrones de reversión adicionales
        patrones_reversa = _detect_patrones_reversa(df_3m)

        # ==========================================================
        # ⏱️ Delay estimado (diferencia entre 3m y 15m)
        # ==========================================================
        delay_estimado = round(abs((rsi_15m[-1] - rsi_3m[-1])) / 10, 2)
        
        # Calcular score final con sistema configurado
        score = analyzer.scoring_system.calculate_score(metrics)
        score = max(0.0, min(score, 1.0))

        # Construir resultado
        resultado = {
            "valido": score >= analyzer.scoring_system.thresholds['min_score'] or modo_rebote,
            "etapa": etapa,
            "etapa_3m": etapa_3m,
            "etapa_5m": etapa_5m,
            "reversa_ult2": reversa_ult2,
            "razon": razon,
            "score": round(score, 2),
            "modo_rebote": modo_rebote,
            "delay_estimado": delay_estimado,
            "rsi_3m": round(rsi_3m[-1], 2) if len(rsi_3m) > 0 else 50.0,
            "rsi_5m": round(rsi_5m[-1], 2) if len(rsi_5m) > 0 else 50.0,
            "rsi_15m": round(rsi_15m[-1], 2) if len(rsi_15m) > 0 else 50.0,
            "macd_3m": round(macd_hist_3m[-1], 5) if len(macd_hist_3m) > 0 else 0.0,
            "macd_5m": round(macd_hist_5m[-1], 5) if len(macd_hist_5m) > 0 else 0.0,
            "tendencia_3m": tendencia_3m,
            "tendencia_5m": tendencia_5m,
            "tendencia_15m": tendencia_15m,
            "pullback_saludable": pullback_saludable,
            "patrones_reversa": patrones_reversa,
            "contexto": analyzer.get_context()
        }

        # No permitir operación contraria a la tendencia mayor (excepto rebotes)
        if direccion_opuesta and not modo_rebote:
            resultado.update({
                "valido": False,
                "razon": f"Operación {direction.upper()} en tendencia mayor {predominante} sin rebote técnico"
            })

        # Actualizar estado del analizador
        analyzer.update_state(resultado)
        
        return resultado

    except KeyError as e:
        logger.error(f"Columna faltante en datos: {e}")
        return {"valido": False, "etapa": "error", "razon": f"Columna faltante: {e}", "score": 0.0}
    except ValueError as e:
        logger.error(f"Valor inválido en cálculo: {e}")
        return {"valido": False, "etapa": "error", "razon": f"Valor inválido: {e}", "score": 0.0}
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        return {"valido": False, "etapa": "error", "razon": f"Error inesperado: {e}", "score": 0.0}

# ===============================================================
# 🔁 Alias de compatibilidad
# ===============================================================
evaluar_timing_multicapa = evaluar_timing_multilayer

# ===============================================================
# 🚀 Ejemplo de uso
# ===============================================================

if __name__ == "__main__":
    # Ejemplo con datos simulados
    print("🧪 Ejecutando ejemplo de uso...")
    
    # Generar datos de prueba
    np.random.seed(42)
    n_candles = 100
    base_price = 50000
    time_index = pd.date_range(start="2025-01-01", periods=n_candles, freq="3min")
    
    # Datos alcistas
    trend = np.linspace(0, 0.05, n_candles)
    noise = np.random.normal(0, 0.01, n_candles)
    prices = base_price * (1 + trend + noise)
    
    df = pd.DataFrame({
        'timestamp': time_index,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_candles))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_candles))),
        'close': prices,
        'volume': np.random.randint(100, 1000, n_candles)
    })
    
    # Asegurar consistencia OHLC
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    
    # Ejecutar análisis
    result = evaluar_timing_multilayer(df, df, df, "long")
    
    print("\n📈 RESULTADOS:")
    print(f"Válido: {result['valido']}")
    print(f"Score: {result['score']}")
    print(f"Etapa: {result['etapa']}")
    print(f"Razón: {result['razon']}")
    print(f"Tendencias: {result['tendencia_3m']}/{result['tendencia_5m']}/{result['tendencia_15m']}")
    print(f"RSI: {result['rsi_3m']}/{result['rsi_5m']}/{result['rsi_15m']}")
    print(f"MACD: {result['macd_3m']}/{result['macd_5m']}")