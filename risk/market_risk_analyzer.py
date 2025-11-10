# market_risk_analyzer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from risk.risk_config import RiskConfig, RiskLevel, MarketRegime

class MarketRiskAnalyzer:
    """Analiza condiciones generales del mercado"""
    
    def __init__(self, config: RiskConfig, binance_client, logger):
        self.config = config
        self.client = binance_client
        self.logger = logger
        self.market_regime = MarketRegime.NORMAL
        
    async def analyze_market_risk(self, symbol: str, signal_type: str) -> Dict[str, Any]:
        """Analiza condiciones de mercado para el símbolo"""
        risk_score = 0
        recommendations = []
        
        try:
            # 1. Determinar régimen de mercado
            market_regime = await self._determine_market_regime(symbol)
            risk_score += market_regime['score']
            recommendations.extend(market_regime['recommendations'])
            
            # 2. Analizar volatilidad
            volatility_risk = await self._analyze_volatility_risk(symbol)
            risk_score += volatility_risk['score']
            recommendations.extend(volatility_risk['recommendations'])
            
            # 3. Verificar liquidez
            liquidity_risk = await self._analyze_liquidity_risk(symbol)
            risk_score += liquidity_risk['score']
            recommendations.extend(liquidity_risk['recommendations'])
            
            # 4. Analizar tendencia y momentum
            trend_risk = await self._analyze_trend_risk(symbol, signal_type)
            risk_score += trend_risk['score']
            recommendations.extend(trend_risk['recommendations'])
            
            # 5. Verificar eventos macro
            event_risk = await self._check_market_events()
            risk_score += event_risk['score']
            recommendations.extend(event_risk['recommendations'])
            
            approved = risk_score <= self.config.MAX_MARKET_RISK_SCORE
            
            return {
                'approved': approved,
                'score': risk_score,
                'market_regime': market_regime['regime'],
                'volatility_risk': volatility_risk,
                'liquidity_risk': liquidity_risk,
                'trend_risk': trend_risk,
                'event_risk': event_risk,
                'recommendations': recommendations,
                'confidence': 1 - (risk_score / 10)
            }
            
        except Exception as e:
            self.logger.error(f"Error en análisis de mercado: {e}")
            return {
                'approved': False,
                'score': 10,
                'recommendations': ['Error en análisis de mercado'],
                'error': str(e)
            }
    
    async def _determine_market_regime(self, symbol: str) -> Dict[str, Any]:
        """Determina el régimen actual del mercado"""
        try:
            # Analizar múltiples timeframe
            volatility = await self._get_current_volatility(symbol)
            trend_strength = await self._calculate_trend_strength(symbol)
            
            if volatility > self.config.VOLATILITY_EXTREME:
                regime = MarketRegime.CRISIS
                score = 3
                recommendation = "Mercado en crisis - alta volatilidad"
            elif volatility > self.config.VOLATILITY_HIGH:
                regime = MarketRegime.VOLATILE
                score = 2
                recommendation = "Mercado volátil - cuidado"
            elif trend_strength > 0.7:
                regime = MarketRegime.TRENDING
                score = 0
                recommendation = "Mercado en tendencia fuerte"
            else:
                regime = MarketRegime.NORMAL
                score = 0
                recommendation = "Mercado normal"
            
            self.market_regime = regime
            
            return {
                'regime': regime,
                'score': score,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'recommendations': [recommendation]
            }
            
        except Exception as e:
            self.logger.warning(f"Error determinando régimen: {e}")
            return {
                'regime': MarketRegime.NORMAL,
                'score': 1,
                'recommendations': ['No se pudo determinar régimen de mercado']
            }
    
    async def _analyze_volatility_risk(self, symbol: str) -> Dict[str, Any]:
        """Analiza riesgo específico de volatilidad"""
        current_vol = await self._get_current_volatility(symbol)
        historical_vol = await self._get_historical_volatility(symbol)
        
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
        
        if vol_ratio > 2.0:
            return {
                'level': RiskLevel.EXTREME,
                'score': 3,
                'current_volatility': current_vol,
                'historical_volatility': historical_vol,
                'vol_ratio': vol_ratio,
                'recommendations': ['Volatilidad extremadamente alta']
            }
        elif vol_ratio > 1.5:
            return {
                'level': RiskLevel.HIGH,
                'score': 2,
                'current_volatility': current_vol,
                'historical_volatility': historical_vol,
                'vol_ratio': vol_ratio,
                'recommendations': ['Volatilidad alta']
            }
        elif vol_ratio > 1.2:
            return {
                'level': RiskLevel.MEDIUM,
                'score': 1,
                'current_volatility': current_vol,
                'historical_volatility': historical_vol,
                'vol_ratio': vol_ratio,
                'recommendations': ['Volatilidad elevada']
            }
        else:
            return {
                'level': RiskLevel.LOW,
                'score': 0,
                'current_volatility': current_vol,
                'historical_volatility': historical_vol,
                'vol_ratio': vol_ratio,
                'recommendations': ['Volatilidad normal']
            }
    
    async def _analyze_liquidity_risk(self, symbol: str) -> Dict[str, Any]:
        """Analiza riesgo de liquidez"""
        try:
            # Obtener depth del order book
            depth = await self.client.get_order_book(symbol=symbol, limit=10)
            
            bids = depth['bids']
            asks = depth['asks']
            
            # Calcular spread
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = (best_ask - best_bid) / best_bid
            
            # Calcular depth en USD (primeros 5 niveles)
            bid_depth = sum(float(bid[0]) * float(bid[1]) for bid in bids[:5])
            ask_depth = sum(float(ask[0]) * float(ask[1]) for ask in asks[:5])
            avg_depth = (bid_depth + ask_depth) / 2
            
            risk_score = 0
            recommendations = []
            
            if spread > 0.002:  # 0.2%
                risk_score += 2
                recommendations.append(f"Spread alto: {spread:.3%}")
            elif spread > 0.001:  # 0.1%
                risk_score += 1
                recommendations.append(f"Spread moderado: {spread:.3%}")
            
            if avg_depth < 5000:  # USD
                risk_score += 2
                recommendations.append(f"Profundidad baja: ${avg_depth:,.0f}")
            elif avg_depth < 20000:
                risk_score += 1
                recommendations.append(f"Profundidad moderada: ${avg_depth:,.0f}")
            
            return {
                'score': risk_score,
                'spread': spread,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'avg_depth': avg_depth,
                'recommendations': recommendations,
                'level': RiskLevel.HIGH if risk_score >= 3 else RiskLevel.MEDIUM if risk_score >= 1 else RiskLevel.LOW
            }
            
        except Exception as e:
            self.logger.warning(f"Error analizando liquidez: {e}")
            return {
                'score': 2,
                'recommendations': ['Error analizando liquidez'],
                'level': RiskLevel.MEDIUM
            }
    
    async def _analyze_trend_risk(self, symbol: str, signal_type: str) -> Dict[str, Any]:
        """Analiza riesgo de tendencia para la señal específica"""
        try:
            # Obtener datos para múltiples timeframes
            trends = {}
            for timeframe in ['15m', '1h', '4h', '1d']:
                trends[timeframe] = await self._calculate_trend_direction(symbol, timeframe)
            
            # Contar tendencias alcistas vs bajistas
            bullish = sum(1 for t in trends.values() if t > 0)
            bearish = sum(1 for t in trends.values() if t < 0)
            
            trend_consensus = bullish - bearish
            risk_score = 0
            recommendations = []
            
            # Evaluar congruencia de señal con tendencia
            if signal_type in ['momentum', 'breakout']:
                # Señales de momentum deberían seguir la tendencia
                if trend_consensus < -2:  # Fuerte tendencia bajista
                    risk_score += 2
                    recommendations.append("Señal de momentum contra tendencia bajista")
            elif signal_type == 'mean_reversion':
                # Mean reversion funciona mejor en rangos
                if abs(trend_consensus) > 2:  # Tendencia muy fuerte
                    risk_score += 1
                    recommendations.append("Mean reversion en mercado con tendencia fuerte")
            
            return {
                'score': risk_score,
                'trend_consensus': trend_consensus,
                'timeframe_trends': trends,
                'recommendations': recommendations,
                'level': RiskLevel.HIGH if risk_score >= 2 else RiskLevel.MEDIUM if risk_score >= 1 else RiskLevel.LOW
            }
            
        except Exception as e:
            self.logger.warning(f"Error analizando tendencia: {e}")
            return {'score': 0, 'recommendations': [], 'level': RiskLevel.LOW}
    
    async def _check_market_events(self) -> Dict[str, Any]:
        """Verifica eventos macroeconómicos o de noticias"""
        # Esta es una implementación simplificada
        # En producción, integrar con fuentes de noticias o calendarios económicos
        
        risk_score = 0
        recommendations = []
        
        # Verificar si es fin de semana (menos liquidez en crypto)
        if datetime.now().weekday() in [5, 6]:  # Sábado o Domingo
            risk_score += 1
            recommendations.append("Mercado de fin de semana - liquidez reducida")
        
        # Verificar horarios de alta volatilidad (open/close US markets)
        utc_now = datetime.utcnow()
        if utc_now.hour in [13, 14, 20, 21]:  # Open/close US markets
            risk_score += 1
            recommendations.append("Horario de alta volatilidad (open/close US)")
        
        return {
            'score': risk_score,
            'recommendations': recommendations,
            'level': RiskLevel.MEDIUM if risk_score >= 1 else RiskLevel.LOW
        }
    
    async def _calculate_trend_strength(self, symbol: str, timeframe: str = "1d") -> float:
        """Calcula fuerza de la tendencia usando ADX"""
        try:
            # Obtener datos OHLCV
            klines = await self.client.get_historical_klines(
                symbol, timeframe, "50 day ago UTC"
            )
            
            # Calcular ADX simplificado
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            closes = [float(k[4]) for k in klines]
            
            # Cálculo simplificado de fuerza de tendencia
            if len(closes) < 20:
                return 0.5
            
            sma_20 = np.mean(closes[-20:])
            price_vs_sma = abs(closes[-1] - sma_20) / sma_20
            
            return min(1.0, price_vs_sma / 0.1)  # Normalizar
            
        except Exception as e:
            self.logger.warning(f"Error calculando fuerza de tendencia: {e}")
            return 0.5
    
    async def _calculate_trend_direction(self, symbol: str, timeframe: str) -> float:
        """Calcula dirección de tendencia (-1 a 1)"""
        try:
            klines = await self.client.get_historical_klines(
                symbol, timeframe, "20 day ago UTC"
            )
            
            closes = [float(k[4]) for k in klines]
            
            if len(closes) < 10:
                return 0
            
            # EMA rápida vs lenta
            ema_fast = self._ema(closes, 9)
            ema_slow = self._ema(closes, 21)
            
            if ema_fast > ema_slow:
                return 1  # Alcista
            else:
                return -1  # Bajista
                
        except Exception as e:
            self.logger.warning(f"Error calculando dirección: {e}")
            return 0
    
    def _ema(self, prices: List[float], period: int) -> float:
        """Calcula EMA"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema