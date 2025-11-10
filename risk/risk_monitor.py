# risk_monitor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from risk.risk_config import RiskConfig, RiskLevel

class RiskMonitor:
    """Monitor en tiempo real de métricas de riesgo"""
    
    def __init__(self, config: RiskConfig, binance_client, logger):
        self.config = config
        self.client = binance_client
        self.logger = logger
        self.trade_history = []
        self.risk_metrics_history = []
        
    async def calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        """Calcula métricas completas de riesgo"""
        try:
            metrics = {
                'timestamp': datetime.now(),
                'portfolio_metrics': await self._calculate_portfolio_metrics(),
                'var_metrics': await self._calculate_var_metrics(),
                'drawdown_analysis': await self._calculate_drawdown_analysis(),
                'performance_metrics': await self._calculate_performance_metrics(),
                'concentration_metrics': await self._calculate_concentration_metrics(),
                'alert_status': await self._check_risk_alerts()
            }
            
            # Guardar en historial
            self.risk_metrics_history.append(metrics)
            
            # Mantener solo últimas 1000 entradas
            if len(self.risk_metrics_history) > 1000:
                self.risk_metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculando métricas: {e}")
            return {'error': str(e)}
    
    async def _calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calcula métricas del portafolio"""
        try:
            positions = await self._get_current_positions()
            total_value = sum(positions.values())
            
            if total_value == 0:
                return {
                    'total_value': 0,
                    'num_positions': 0,
                    'avg_position_size': 0,
                    'largest_position': 0,
                    'diversification_score': 0
                }
            
            position_sizes = list(positions.values())
            avg_position = np.mean(position_sizes)
            largest_position = max(position_sizes)
            
            # Calcular score de diversificación (0-1, 1 = perfectamente diversificado)
            if len(position_sizes) <= 1:
                diversification = 0
            else:
                # Usar índice Herfindahl
                market_shares = [size / total_value for size in position_sizes]
                herfindahl = sum(share ** 2 for share in market_shares)
                diversification = 1 - herfindahl
            
            return {
                'total_value': total_value,
                'num_positions': len(positions),
                'avg_position_size': avg_position,
                'largest_position': largest_position,
                'largest_position_ratio': largest_position / total_value,
                'diversification_score': diversification
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculando métricas de portafolio: {e}")
            return {}
    
    async def _calculate_var_metrics(self, confidence_levels: List[float] = None) -> Dict[str, Any]:
        """Calcula Value at Risk para diferentes niveles de confianza"""
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        
        try:
            # Obtener retornos históricos del portafolio
            portfolio_returns = await self._get_portfolio_returns(period=90)  # 90 días
            
            if len(portfolio_returns) < 10:
                return {'error': 'Datos insuficientes'}
            
            var_metrics = {}
            
            for confidence in confidence_levels:
                # VaR histórico
                var_historical = np.percentile(portfolio_returns, (1 - confidence) * 100)
                
                # VaR paramétrico (asumiendo distribución normal)
                mean_return = np.mean(portfolio_returns)
                std_return = np.std(portfolio_returns)
                var_parametric = mean_return + std_return * np.percentile(np.random.standard_normal(10000), (1 - confidence) * 100)
                
                var_metrics[f'var_{int(confidence*100)}'] = {
                    'historical': var_historical,
                    'parametric': var_parametric,
                    'confidence': confidence
                }
            
            # Expected Shortfall (CVaR)
            es_95 = np.mean([r for r in portfolio_returns if r <= var_metrics['var_95']['historical']])
            
            return {
                **var_metrics,
                'expected_shortfall_95': es_95,
                'mean_return': np.mean(portfolio_returns),
                'std_return': std_return
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculando VaR: {e}")
            return {'error': str(e)}
    
    async def _calculate_drawdown_analysis(self) -> Dict[str, Any]:
        """Analiza drawdowns del portafolio"""
        try:
            portfolio_values = await self._get_portfolio_value_history(period=90)
            
            if len(portfolio_values) < 2:
                return {'error': 'Datos insuficientes'}
            
            # Calcular drawdowns
            peak = portfolio_values[0]
            drawdowns = []
            max_drawdown = 0
            drawdown_duration = 0
            current_drawdown_duration = 0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                    current_drawdown_duration = 0
                else:
                    drawdown = (peak - value) / peak
                    drawdowns.append(drawdown)
                    max_drawdown = max(max_drawdown, drawdown)
                    current_drawdown_duration += 1
                    drawdown_duration = max(drawdown_duration, current_drawdown_duration)
            
            current_drawdown = (peak - portfolio_values[-1]) / peak if peak > 0 else 0
            
            return {
                'max_drawdown': max_drawdown,
                'current_drawdown': current_drawdown,
                'avg_drawdown': np.mean(drawdowns) if drawdowns else 0,
                'max_drawdown_duration': drawdown_duration,
                'recovery_factor': -np.mean(portfolio_values) / max_drawdown if max_drawdown > 0 else 0
            }
            
        except Exception as e:
            self.logger.warning(f"Error analizando drawdown: {e}")
            return {'error': str(e)}
    
    async def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calcula métricas de performance ajustadas por riesgo"""
        try:
            portfolio_returns = await self._get_portfolio_returns(period=90)
            
            if len(portfolio_returns) < 10:
                return {'error': 'Datos insuficientes'}
            
            mean_return = np.mean(portfolio_returns)
            std_return = np.std(portfolio_returns)
            negative_returns = [r for r in portfolio_returns if r < 0]
            std_negative = np.std(negative_returns) if negative_returns else 0
            
            # Sharpe Ratio (asumiendo risk-free rate = 0 para crypto)
            sharpe = mean_return / std_return if std_return > 0 else 0
            
            # Sortino Ratio
            sortino = mean_return / std_negative if std_negative > 0 else 0
            
            # Calmar Ratio
            drawdown_analysis = await self._calculate_drawdown_analysis()
            max_drawdown = drawdown_analysis.get('max_drawdown', 0.01)
            calmar = mean_return / max_drawdown if max_drawdown > 0 else 0
            
            # Win rate
            winning_trades = len([r for r in portfolio_returns if r > 0])
            total_trades = len(portfolio_returns)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            return {
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'win_rate': win_rate,
                'profit_factor': -np.sum([r for r in portfolio_returns if r > 0]) / np.sum([r for r in portfolio_returns if r < 0]) if any(r < 0 for r in portfolio_returns) else float('inf'),
                'avg_win': np.mean([r for r in portfolio_returns if r > 0]) if any(r > 0 for r in portfolio_returns) else 0,
                'avg_loss': np.mean([r for r in portfolio_returns if r < 0]) if any(r < 0 for r in portfolio_returns) else 0
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculando métricas de performance: {e}")
            return {'error': str(e)}
    
    async def _calculate_concentration_metrics(self) -> Dict[str, Any]:
        """Calcula métricas de concentración del portafolio"""
        try:
            positions = await self._get_current_positions()
            position_values = list(positions.values())
            total_value = sum(position_values)
            
            if total_value == 0:
                return {
                    'gini_coefficient': 0,
                    'herfindahl_index': 0,
                    'top_3_concentration': 0,
                    'effective_n': 0
                }
            
            # Coeficiente de Gini
            sorted_values = sorted(position_values)
            n = len(sorted_values)
            gini = 1 - 2 * sum((i + 1) * x for i, x in enumerate(sorted_values)) / (n * sum(sorted_values))
            
            # Índice Herfindahl
            shares = [v / total_value for v in position_values]
            herfindahl = sum(share ** 2 for share in shares)
            
            # Concentración top 3
            top_3 = sum(sorted(position_values, reverse=True)[:3]) / total_value
            
            # Número efectivo de posiciones
            effective_n = 1 / herfindahl if herfindahl > 0 else 0
            
            return {
                'gini_coefficient': gini,
                'herfindahl_index': herfindahl,
                'top_3_concentration': top_3,
                'effective_n': effective_n,
                'concentration_level': 'HIGH' if herfindahl > 0.25 else 'MEDIUM' if herfindahl > 0.15 else 'LOW'
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculando concentración: {e}")
            return {'error': str(e)}
    
    async def _check_risk_alerts(self) -> Dict[str, Any]:
        """Verifica condiciones que requieren alertas"""
        alerts = []
        
        try:
            # Verificar drawdown actual
            drawdown = await self._calculate_drawdown_analysis()
            current_dd = drawdown.get('current_drawdown', 0)
            if current_dd > self.config.MAX_TRADE_DRAWDOWN:
                alerts.append({
                    'level': 'HIGH',
                    'message': f'Drawdown actual excede límite: {current_dd:.2%}',
                    'metric': 'drawdown',
                    'value': current_dd
                })
            
            # Verificar concentración
            concentration = await self._calculate_concentration_metrics()
            herfindahl = concentration.get('herfindahl_index', 0)
            if herfindahl > 0.3:
                alerts.append({
                    'level': 'MEDIUM',
                    'message': f'Portafolio muy concentrado: {herfindahl:.3f}',
                    'metric': 'concentration',
                    'value': herfindahl
                })
            
            # Verificar pérdidas consecutivas
            if len(self.trade_history) >= 3:
                recent_trades = self.trade_history[-3:]
                consecutive_losses = sum(1 for trade in recent_trades if trade.get('pnl', 0) < 0)
                if consecutive_losses >= 3:
                    alerts.append({
                        'level': 'MEDIUM',
                        'message': f'{consecutive_losses} pérdidas consecutivas',
                        'metric': 'consecutive_losses',
                        'value': consecutive_losses
                    })
            
            return {
                'active_alerts': alerts,
                'alert_count': len(alerts),
                'highest_alert': max([alert['level'] for alert in alerts], key=lambda x: ['LOW', 'MEDIUM', 'HIGH'].index(x)) if alerts else 'LOW'
            }
            
        except Exception as e:
            self.logger.warning(f"Error verificando alertas: {e}")
            return {'active_alerts': [], 'alert_count': 0, 'highest_alert': 'LOW'}
    
    async def _get_current_positions(self) -> Dict[str, float]:
        """Obtiene posiciones actuales"""
        # Implementación similar a la de PortfolioManager
        # Por simplicidad, retornar diccionario vacío
        return {}
    
    async def _get_portfolio_returns(self, period: int = 90) -> List[float]:
        """Obtiene retornos históricos del portafolio"""
        # En implementación real, obtener de base de datos o API
        # Por ahora, retornar datos de ejemplo
        return np.random.normal(0.001, 0.02, 60).tolist()  # 60 retornos diarios
    
    async def _get_portfolio_value_history(self, period: int = 90) -> List[float]:
        """Obtiene historial de valores del portafolio"""
        # En implementación real, obtener de base de datos
        # Por ahora, retornar datos de ejemplo
        base_value = 10000
        returns = await self._get_portfolio_returns(period)
        values = [base_value]
        
        for ret in returns:
            values.append(values[-1] * (1 + ret))
        
        return values