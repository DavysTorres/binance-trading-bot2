# position_sizing.py
import numpy as np
from typing import Dict, Any
from risk.risk_config import RiskConfig

class PositionSizing:
    """Gestiona cálculo de tamaño de posición y leverage"""
    
    def __init__(self, config: RiskConfig, binance_client, logger):
        self.config = config
        self.client = binance_client
        self.logger = logger
        
    async def calculate_position_risk(self, symbol: str, signal_type: str, 
                                    proposed_size: float, leverage: float) -> Dict[str, Any]:
        """Calcula riesgo de posición y tamaño óptimo"""
        risk_score = 0
        recommendations = []
        
        try:
            # 1. Calcular tamaño base
            base_size = await self._calculate_base_position_size(signal_type)
            
            # 2. Ajustar por volatilidad
            volatility_adjustment = await self._adjust_for_volatility(symbol)
            
            # 3. Ajustar por calidad de señal
            signal_adjustment = await self._adjust_for_signal_quality(signal_type)
            
            # 4. Calcular leverage máximo
            max_leverage = await self._calculate_max_leverage(symbol, signal_type)
            effective_leverage = min(leverage, max_leverage)
            
            # 5. Calcular tamaño final
            recommended_size = base_size * volatility_adjustment * signal_adjustment
            
            # 6. Aplicar límites absolutos
            recommended_size = max(self.config.MIN_POSITION_SIZE, 
                                 min(recommended_size, self.config.MAX_POSITION_SIZE))
            
            # 7. Evaluar riesgo del tamaño propuesto
            size_ratio = proposed_size / recommended_size if recommended_size > 0 else 0
            
            if size_ratio > 1.5:
                risk_score += 4
                recommendations.append(f"Tamaño propuesto muy grande: {size_ratio:.1f}x recomendado")
            elif size_ratio > 1.2:
                risk_score += 2
                recommendations.append(f"Tamaño propuesto grande: {size_ratio:.1f}x recomendado")
            elif size_ratio < 0.5:
                risk_score += 1
                recommendations.append(f"Tamaño propuesto muy pequeño: {size_ratio:.1f}x recomendado")
            
            # 8. Evaluar riesgo de leverage
            leverage_risk = await self._assess_leverage_risk(effective_leverage, symbol)
            risk_score += leverage_risk['score']
            recommendations.extend(leverage_risk['recommendations'])
            
            approved = risk_score <= 3 and effective_leverage <= max_leverage
            
            return {
                'approved': approved,
                'score': risk_score,
                'recommended_size': recommended_size,
                'approved_size': recommended_size if approved else min(recommended_size, proposed_size),
                'max_leverage': max_leverage,
                'effective_leverage': effective_leverage,
                'base_size': base_size,
                'volatility_adjustment': volatility_adjustment,
                'signal_adjustment': signal_adjustment,
                'size_ratio': size_ratio,
                'recommendations': recommendations,
                'leverage_risk': leverage_risk
            }
            
        except Exception as e:
            self.logger.error(f"Error en cálculo de posición: {e}")
            return {
                'approved': False,
                'score': 10,
                'recommended_size': 0,
                'approved_size': 0,
                'max_leverage': 1,
                'error': str(e)
            }
    
    async def _calculate_base_position_size(self, signal_type: str) -> float:
        """Calcula tamaño base usando Kelly Criterion o riesgo fijo"""
        if self.config.USE_KELLY_CRITERION:
            win_rate = self.config.STRATEGY_WIN_RATES.get(signal_type, 0.5)
            win_loss_ratio = self.config.AVG_WIN_LOSS_RATIO.get(signal_type, 1.0)
            
            # Kelly Criterion: f = p - (1-p)/b
            kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
            
            # Fractional Kelly para ser conservador
            fractional_kelly = kelly_fraction * self.config.KELLY_FRACTION
            
            # Obtener capital total
            total_capital = await self._get_total_capital()
            
            return max(self.config.MIN_POSITION_SIZE, 
                      min(fractional_kelly * total_capital, self.config.MAX_POSITION_SIZE))
        else:
            # Usar riesgo fijo
            total_capital = await self._get_total_capital()
            fixed_risk = self.config.FIXED_RISK_PER_TRADE * total_capital
            return max(self.config.MIN_POSITION_SIZE, 
                      min(fixed_risk, self.config.MAX_POSITION_SIZE))
    
    async def _adjust_for_volatility(self, symbol: str) -> float:
        """Ajusta posición basado en volatilidad actual"""
        try:
            current_volatility = await self._get_current_volatility(symbol)
            historical_volatility = await self._get_historical_volatility(symbol)
            
            if historical_volatility == 0:
                return 1.0
            
            vol_ratio = current_volatility / historical_volatility
            
            if vol_ratio > 2.0:
                return 0.3  # Reducir 70%
            elif vol_ratio > 1.5:
                return 0.5  # Reducir 50%
            elif vol_ratio > 1.2:
                return 0.8  # Reducir 20%
            else:
                return 1.0  # Sin ajuste
                
        except Exception as e:
            self.logger.warning(f"Error calculando ajuste por volatilidad: {e}")
            return 1.0
    
    async def _adjust_for_signal_quality(self, signal_type: str) -> float:
        """Ajusta posición basado en calidad de señal"""
        # Basado en win rate histórico de la estrategia
        win_rate = self.config.STRATEGY_WIN_RATES.get(signal_type, 0.5)
        
        if win_rate > 0.6:
            return 1.2  # Aumentar 20%
        elif win_rate > 0.55:
            return 1.1  # Aumentar 10%
        elif win_rate < 0.45:
            return 0.7  # Reducir 30%
        elif win_rate < 0.5:
            return 0.8  # Reducir 20%
        else:
            return 1.0
    
    async def _calculate_max_leverage(self, symbol: str, signal_type: str) -> float:
        """Calcula leverage máximo permitido"""
        base_leverage = 1.0
        
        # Ajustar por tipo de estrategia
        if signal_type in ['scalping', 'arbitrage']:
            base_leverage = 2.0
        elif signal_type in ['momentum', 'breakout']:
            base_leverage = 3.0
        elif signal_type == 'mean_reversion':
            base_leverage = 1.5
        
        # Reducir leverage en alta volatilidad
        volatility = await self._get_current_volatility(symbol)
        if volatility > self.config.VOLATILITY_EXTREME:
            base_leverage *= 0.5
        elif volatility > self.config.VOLATILITY_HIGH:
            base_leverage *= 0.7
        
        return min(base_leverage, self.config.MAX_LEVERAGE_OVERALL)
    
    async def _assess_leverage_risk(self, leverage: float, symbol: str) -> Dict[str, Any]:
        """Evalúa riesgo asociado al leverage"""
        risk_score = 0
        recommendations = []
        
        if leverage > 5.0:
            risk_score += 4
            recommendations.append("Leverage extremadamente alto")
        elif leverage > 3.0:
            risk_score += 2
            recommendations.append("Leverage alto")
        elif leverage > 1.0:
            risk_score += 1
        
        # Considerar volatilidad del símbolo
        volatility = await self._get_current_volatility(symbol)
        if volatility > self.config.VOLATILITY_HIGH and leverage > 1.0:
            risk_score += 2
            recommendations.append("Alto leverage en mercado volátil")
        
        return {
            'score': risk_score,
            'recommendations': recommendations,
            'leverage': leverage
        }
    
    async def _get_current_volatility(self, symbol: str, days: int = 10) -> float:
        """Calcula volatilidad actual (desviación estándar de retornos diarios)"""
        try:
            klines = await self.client.get_historical_klines(
                symbol, "1d", f"{days} day ago UTC"
            )
            
            closes = [float(k[4]) for k in klines]
            returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
            
            return np.std(returns) if returns else 0.02  # Default 2%
            
        except Exception as e:
            self.logger.warning(f"Error calculando volatilidad: {e}")
            return 0.02
    
    async def _get_historical_volatility(self, symbol: str, days: int = 30) -> float:
        """Obtiene volatilidad histórica"""
        return await self._get_current_volatility(symbol, days)
    
    async def _get_total_capital(self) -> float:
        """Obtiene capital total disponible"""
        try:
            account = await self.client.get_account()
            total = 0
            
            for balance in account['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                
                if asset == 'USDT':
                    total += free + locked
                elif free + locked > 0:
                    # Convertir a USDT
                    symbol = f"{asset}USDT"
                    try:
                        ticker = await self.client.get_symbol_ticker(symbol=symbol)
                        price = float(ticker['price'])
                        total += (free + locked) * price
                    except:
                        continue
            
            return total
            
        except Exception as e:
            self.logger.error(f"Error obteniendo capital total: {e}")
            return 1000  # Fallback value