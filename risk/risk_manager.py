# risk_manager.py
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
from risk.risk_config import RiskConfig
from risk.portfolio_manager import PortfolioManager
from risk.position_sizing import PositionSizing
from risk.market_risk_analyzer import MarketRiskAnalyzer
from risk.execution_risk import ExecutionRiskManager
from risk.risk_monitor import RiskMonitor


@dataclass
class RiskAssessment:
    approved: bool
    risk_score: float
    position_size: float
    max_leverage: float
    risk_factors: Dict[str, Any]
    recommendations: List[str]
    confidence: float

class RiskManager:
    """Sistema principal de gestión de riesgo"""
    
    def __init__(self, config: RiskConfig, binance_client, logger=None):
        self.config = config
        self.client = binance_client
        self.logger = logger or logging.getLogger(__name__)
        
        # Inicializar submódulos
        self.portfolio = PortfolioManager(config, binance_client, logger)
        self.position = PositionSizing(config, binance_client, logger)
        self.market = MarketRiskAnalyzer(config, binance_client, logger)
        self.execution = ExecutionRiskManager(config, binance_client, logger)
        self.monitor = RiskMonitor(config, binance_client, logger)
        
        # Estado del sistema
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.consecutive_losses = 0
        self.last_trade_time = None
        
    async def evaluate_trade(self, symbol: str, signal_type: str, 
                           proposed_size: float, leverage: float = 1.0) -> RiskAssessment:
        """
        Evaluación completa de riesgo para una operación propuesta
        """
        self.logger.info(f"Evaluando riesgo para {symbol} - {signal_type}")
        
        risk_factors = {}
        recommendations = []
        total_risk_score = 0.0
        
        try:
            # 1. Verificación de límites globales
            global_risk = await self._check_global_limits()
            if not global_risk['approved']:
                return RiskAssessment(
                    approved=False,
                    risk_score=999,
                    position_size=0,
                    max_leverage=0,
                    risk_factors={'global': global_risk},
                    recommendations=global_risk['recommendations'],
                    confidence=0.0
                )
            risk_factors['global'] = global_risk
            total_risk_score += global_risk['score']
            
            # 2. Análisis de portafolio
            portfolio_risk = await self.portfolio.analyze_portfolio_risk(symbol, signal_type)
            risk_factors['portfolio'] = portfolio_risk
            total_risk_score += portfolio_risk['score']
            
            if not portfolio_risk['approved']:
                recommendations.extend(portfolio_risk['recommendations'])
            
            # 3. Cálculo de tamaño de posición
            position_risk = await self.position.calculate_position_risk(
                symbol, signal_type, proposed_size, leverage
            )
            risk_factors['position'] = position_risk
            total_risk_score += position_risk['score']
            
            if not position_risk['approved']:
                recommendations.extend(position_risk['recommendations'])
            
            # 4. Análisis de condiciones de mercado
            market_risk = await self.market.analyze_market_risk(symbol, signal_type)
            risk_factors['market'] = market_risk
            total_risk_score += market_risk['score']
            
            if not market_risk['approved']:
                recommendations.extend(market_risk['recommendations'])
            
            # 5. Verificación de ejecución
            execution_risk = await self.execution.analyze_execution_risk(symbol, proposed_size)
            risk_factors['execution'] = execution_risk
            total_risk_score += execution_risk['score']
            
            if not execution_risk['approved']:
                recommendations.extend(execution_risk['recommendations'])
            
            # 6. Decisión final
            approved = (
                global_risk['approved'] and
                portfolio_risk['approved'] and
                position_risk['approved'] and
                market_risk['approved'] and
                execution_risk['approved'] and
                total_risk_score <= self.config.MAX_RISK_SCORE
            )
            
            # Calcular confianza basada en factores de riesgo
            confidence = self._calculate_confidence(risk_factors)
            
            # Recomendación adicional si está en límite
            if total_risk_score > self.config.MAX_RISK_SCORE * 0.8:
                recommendations.append("Riesgo elevado - considerar reducir tamaño de posición")
            
            assessment = RiskAssessment(
                approved=approved,
                risk_score=total_risk_score,
                position_size=position_risk['approved_size'],
                max_leverage=position_risk['max_leverage'],
                risk_factors=risk_factors,
                recommendations=recommendations,
                confidence=confidence
            )
            
            self.logger.info(f"Evaluación completada: {'APROBADO' if approved else 'RECHAZADO'} - Score: {total_risk_score:.2f}")
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error en evaluación de riesgo: {e}")
            return RiskAssessment(
                approved=False,
                risk_score=999,
                position_size=0,
                max_leverage=0,
                risk_factors={'error': str(e)},
                recommendations=['Error en evaluación de riesgo'],
                confidence=0.0
            )
    
    async def _check_global_limits(self) -> Dict[str, Any]:
        """Verifica límites globales del sistema"""
        risk_score = 0
        recommendations = []
        approved = True
        
        # Verificar pérdidas diarias
        if self.daily_pnl < -self.config.MAX_DAILY_LOSS:
            approved = False
            risk_score += 10
            recommendations.append(f"Pérdida diaria excedida: {self.daily_pnl:.2%}")
        
        # Verificar pérdidas semanales
        if self.weekly_pnl < -self.config.MAX_WEEKLY_LOSS:
            approved = False
            risk_score += 10
            recommendations.append(f"Pérdida semanal excedida: {self.weekly_pnl:.2%}")
        
        # Verificar pérdidas consecutivas
        if self.consecutive_losses >= self.config.MAX_CONSECUTIVE_LOSSES:
            approved = False
            risk_score += 8
            recommendations.append(f"Máximo de pérdidas consecutivas alcanzado: {self.consecutive_losses}")
        
        # Verificar frecuencia de trading
        if self.last_trade_time and (datetime.now() - self.last_trade_time).seconds < 30:
            risk_score += 2
            recommendations.append("Frecuencia de trading muy alta")
        
        return {
            'approved': approved,
            'score': risk_score,
            'recommendations': recommendations,
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'consecutive_losses': self.consecutive_losses
        }
    
    def _calculate_confidence(self, risk_factors: Dict[str, Any]) -> float:
        """Calcula confianza basada en factores de riesgo"""
        confidence = 1.0
        
        for category, factors in risk_factors.items():
            if 'confidence' in factors:
                confidence *= factors['confidence']
            elif 'score' in factors:
                # Convertir score a factor de confianza (0-1)
                max_score = 10  # Score máximo por categoría
                category_confidence = max(0, 1 - (factors['score'] / max_score))
                confidence *= category_confidence
        
        return confidence ** (1/len(risk_factors))  # Media geométrica
    
    async def update_pnl(self, pnl: float):
        """Actualiza PNL y contadores de pérdidas"""
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.monthly_pnl += pnl
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        self.last_trade_time = datetime.now()
    
    async def reset_daily_pnl(self):
        """Resetea PNL diario (ejecutar al inicio del día)"""
        self.daily_pnl = 0.0
    
    async def reset_weekly_pnl(self):
        """Resetea PNL semanal (ejecutar al inicio de la semana)"""
        self.weekly_pnl = 0.0
    
    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas completas de riesgo"""
        return await self.monitor.calculate_comprehensive_metrics()