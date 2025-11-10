# execution_risk.py
import time
from typing import Dict, Any, List
from risk.risk_config import RiskConfig, RiskLevel

class ExecutionRiskManager:
    """Gestiona riesgos relacionados con la ejecución"""
    
    def __init__(self, config: RiskConfig, binance_client, logger):
        self.config = config
        self.client = binance_client
        self.logger = logger
        self.api_errors = 0
        self.last_api_call = time.time()
        
    async def analyze_execution_risk(self, symbol: str, size: float) -> Dict[str, Any]:
        """Analiza riesgos de ejecución"""
        risk_score = 0
        recommendations = []
        
        try:
            # 1. Verificar estado de API
            api_risk = await self._check_api_health()
            risk_score += api_risk['score']
            recommendations.extend(api_risk['recommendations'])
            
            # 2. Verificar límites de orden
            order_risk = await self._check_order_limits(symbol, size)
            risk_score += order_risk['score']
            recommendations.extend(order_risk['recommendations'])
            
            # 3. Estimar slippage
            slippage_risk = await self._estimate_slippage(symbol, size)
            risk_score += slippage_risk['score']
            recommendations.extend(slippage_risk['recommendations'])
            
            # 4. Verificar latencia
            latency_risk = self._check_latency()
            risk_score += latency_risk['score']
            recommendations.extend(latency_risk['recommendations'])
            
            approved = risk_score <= self.config.MAX_EXECUTION_RISK_SCORE
            
            return {
                'approved': approved,
                'score': risk_score,
                'api_risk': api_risk,
                'order_risk': order_risk,
                'slippage_risk': slippage_risk,
                'latency_risk': latency_risk,
                'recommendations': recommendations,
                'confidence': 1 - (risk_score / 10)
            }
            
        except Exception as e:
            self.logger.error(f"Error en análisis de ejecución: {e}")
            return {
                'approved': False,
                'score': 10,
                'recommendations': ['Error en análisis de ejecución'],
                'error': str(e)
            }
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """Verifica salud de la API"""
        risk_score = 0
        recommendations = []
        
        try:
            # Verificar conectividad
            await self.client.ping()
            
            # Verificar weight usage
            # Binance tiene límites de requests por minuto
            current_time = time.time()
            if current_time - self.last_api_call < 0.1:  # 100ms entre calls
                risk_score += 1
                recommendations.append("Frecuencia de API muy alta")
            
            self.last_api_call = current_time
            
            # Resetear contador de errores si la llamada fue exitosa
            self.api_errors = max(0, self.api_errors - 1)
            
        except Exception as e:
            self.api_errors += 1
            risk_score += 3
            recommendations.append(f"Error de API: {e}")
        
        if self.api_errors > 5:
            risk_score += 2
            recommendations.append("Múltiples errores de API consecutivos")
        
        return {
            'score': risk_score,
            'api_errors': self.api_errors,
            'recommendations': recommendations,
            'level': RiskLevel.HIGH if risk_score >= 3 else RiskLevel.MEDIUM if risk_score >= 1 else RiskLevel.LOW
        }
    
    async def _check_order_limits(self, symbol: str, size: float) -> Dict[str, Any]:
        """Verifica límites de orden y lotes"""
        risk_score = 0
        recommendations = []
        
        try:
            # Obtener info del símbolo
            exchange_info = await self.client.get_exchange_info()
            symbol_info = next(s for s in exchange_info['symbols'] if s['symbol'] == symbol)
            
            # Verificar límites de cantidad
            lot_size_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE')
            min_qty = float(lot_size_filter['minQty'])
            max_qty = float(lot_size_filter['maxQty'])
            step_size = float(lot_size_filter['stepSize'])
            
            # Verificar límites de precio
            price_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER')
            min_price = float(price_filter['minPrice'])
            max_price = float(price_filter['maxPrice'])
            tick_size = float(price_filter['tickSize'])
            
            # Obtener precio actual para calcular cantidad
            ticker = await self.client.get_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            quantity = size / price
            
            # Verificar cantidad mínima
            if quantity < min_qty:
                risk_score += 3
                recommendations.append(f"Cantidad por debajo del mínimo: {quantity} < {min_qty}")
            
            # Verificar step size
            if quantity % step_size != 0:
                risk_score += 2
                recommendations.append(f"Cantidad no válida para step size: {step_size}")
            
            return {
                'score': risk_score,
                'min_qty': min_qty,
                'max_qty': max_qty,
                'step_size': step_size,
                'calculated_quantity': quantity,
                'recommendations': recommendations,
                'level': RiskLevel.HIGH if risk_score >= 3 else RiskLevel.MEDIUM if risk_score >= 1 else RiskLevel.LOW
            }
            
        except Exception as e:
            self.logger.warning(f"Error verificando límites: {e}")
            return {
                'score': 2,
                'recommendations': ['Error verificando límites de orden'],
                'level': RiskLevel.MEDIUM
            }
    
    async def _estimate_slippage(self, symbol: str, size: float) -> Dict[str, Any]:
        """Estima slippage potencial para la orden"""
        try:
            # Obtener order book
            depth = await self.client.get_order_book(symbol=symbol, limit=20)
            
            # Calcular slippage para compra
            asks = depth['asks']
            total_cost = 0
            remaining_size = size
            
            for ask in asks:
                price = float(ask[0])
                quantity = float(ask[1])
                available = price * quantity
                
                if available >= remaining_size:
                    total_cost += remaining_size * price
                    remaining_size = 0
                    break
                else:
                    total_cost += available * price
                    remaining_size -= available
            
            if remaining_size > 0:
                # No hay suficiente liquidez en los primeros 20 niveles
                avg_price = total_cost / (size - remaining_size)
                slippage = 0.05  # Estimación conservadora 5%
            else:
                avg_price = total_cost / size
                current_price = float(asks[0][0])
                slippage = (avg_price - current_price) / current_price
            
            risk_score = 0
            recommendations = []
            
            if slippage > 0.01:  # 1%
                risk_score += 3
                recommendations.append(f"Slippage potencial alto: {slippage:.3%}")
            elif slippage > 0.005:  # 0.5%
                risk_score += 2
                recommendations.append(f"Slippage potencial moderado: {slippage:.3%}")
            elif slippage > 0.002:  # 0.2%
                risk_score += 1
                recommendations.append(f"Slippage potencial: {slippage:.3%}")
            
            return {
                'score': risk_score,
                'estimated_slippage': slippage,
                'avg_execution_price': avg_price,
                'recommendations': recommendations,
                'level': RiskLevel.HIGH if risk_score >= 3 else RiskLevel.MEDIUM if risk_score >= 1 else RiskLevel.LOW
            }
            
        except Exception as e:
            self.logger.warning(f"Error estimando slippage: {e}")
            return {
                'score': 1,
                'estimated_slippage': 0.005,
                'recommendations': ['No se pudo estimar slippage accurately'],
                'level': RiskLevel.MEDIUM
            }
    
    def _check_latency(self) -> Dict[str, Any]:
        """Verifica latencia del sistema"""
        risk_score = 0
        recommendations = []
        
        # Verificar tiempo desde última llamada a API
        current_time = time.time()
        latency = current_time - self.last_api_call
        
        if latency > 10:  # 10 segundos
            risk_score += 2
            recommendations.append(f"Alta latencia desde última llamada: {latency:.1f}s")
        elif latency > 5:
            risk_score += 1
            recommendations.append(f"Latencia moderada: {latency:.1f}s")
        
        return {
            'score': risk_score,
            'latency': latency,
            'recommendations': recommendations,
            'level': RiskLevel.MEDIUM if risk_score >= 1 else RiskLevel.LOW
        }