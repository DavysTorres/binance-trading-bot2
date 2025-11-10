# strategy_executor.py
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

class StrategyExecutor:
    """
    Ejecuta y gestiona las estrategias de trading,
    combinando señales y gestionando posiciones
    """
    
    def __init__(self, strategy_manager, risk_manager, binance_client):
        self.strategy_manager = strategy_manager
        self.risk_manager = risk_manager
        self.client = binance_client
        self.logger = logging.getLogger("strategy.executor")
        
        self.active_positions = {}
        self.signal_history = []
        self.performance_tracker = {}
        
    async def execute_strategy_cycle(self, symbol: str, market_data: Dict) -> Dict:
        """Ejecuta un ciclo completo de estrategias para un símbolo"""
        try:
            # 1. Obtener señales combinadas de todas las estrategias
            combined_signal = await self.strategy_manager.get_combined_signals(
                symbol, market_data
            )
            
            # 2. Evaluar riesgo de la operación propuesta
            if combined_signal['action'] in ['BUY', 'SELL']:
                # Calcular tamaño de posición basado en la señal
                proposed_size = self._calculate_position_size(
                    symbol, combined_signal, market_data
                )
                
                # Evaluar riesgo
                risk_assessment = await self.risk_manager.evaluate_trade(
                    symbol, 
                    combined_signal['action'].lower(),
                    proposed_size,
                    leverage=1.0  # Sin leverage inicial
                )
                
                # 3. Ejecutar si está aprobado por riesgo
                if risk_assessment.approved:
                    execution_result = await self._execute_trade(
                        symbol, combined_signal, risk_assessment
                    )
                    
                    return {
                        'executed': True,
                        'signal': combined_signal,
                        'risk_assessment': risk_assessment,
                        'execution_result': execution_result
                    }
                else:
                    self.logger.info(f"Operación rechazada por riesgo: {risk_assessment}")
                    return {
                        'executed': False,
                        'signal': combined_signal,
                        'risk_assessment': risk_assessment,
                        'reason': 'risk_rejection'
                    }
            else:
                return {
                    'executed': False,
                    'signal': combined_signal,
                    'reason': 'no_signal'
                }
                
        except Exception as e:
            self.logger.error(f"Error en ciclo de ejecución: {e}")
            return {
                'executed': False,
                'error': str(e)
            }
    
    def _calculate_position_size(self, symbol: str, signal: Dict, market_data: Dict) -> float:
        """Calcula tamaño de posición basado en la señal y condiciones"""
        base_size = 1000  # Tamaño base en USD
        
        # Ajustar por confianza de la señal
        confidence_multiplier = signal.get('confidence', 0.5)
        adjusted_size = base_size * confidence_multiplier
        
        # Ajustar por volatilidad
        if '1h' in market_data:
            df = market_data['1h']
            if len(df) > 20:
                volatility = df['close'].pct_change().std()
                if volatility > 0.05:  # Alta volatilidad
                    adjusted_size *= 0.5
                elif volatility < 0.01:  # Baja volatilidad
                    adjusted_size *= 1.2
        
        # Ajustar por número de estrategias coincidentes
        strategy_count = signal.get('strategy_count', 1)
        if strategy_count >= 3:
            adjusted_size *= 1.5
        elif strategy_count == 1:
            adjusted_size *= 0.7
        
        return max(100, adjusted_size)  # Mínimo $100
    
    async def _execute_trade(self, symbol: str, signal: Dict, risk_assessment) -> Dict:
        """Ejecuta la operación en el exchange"""
        try:
            action = signal['action']
            size = risk_assessment.position_size
            
            # Determinar tipo de orden
            order_type = 'MARKET'  # Podría ser LIMIT en estrategias específicas
            
            # Ejecutar orden
            if action == 'BUY':
                order_result = await self.client.order_market_buy(
                    symbol=symbol,
                    quantity=self._calculate_quantity(symbol, size)
                )
            elif action == 'SELL':
                order_result = await self.client.order_market_sell(
                    symbol=symbol,
                    quantity=self._calculate_quantity(symbol, size)
                )
            else:
                return {'error': 'Acción no válida'}
            
            # Registrar posición
            self._update_active_positions(symbol, action, order_result, signal)
            
            # Registrar en historial
            trade_record = {
                'symbol': symbol,
                'action': action,
                'size': size,
                'timestamp': datetime.now(),
                'signal': signal,
                'order_id': order_result.get('orderId'),
                'executed_price': float(order_result.get('fills', [{}])[0].get('price', 0))
            }
            
            self.signal_history.append(trade_record)
            
            return {
                'success': True,
                'order_result': order_result,
                'trade_record': trade_record
            }
            
        except Exception as e:
            self.logger.error(f"Error ejecutando orden: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_quantity(self, symbol: str, size_usd: float) -> float:
        """Calcula cantidad basada en tamaño en USD y precio actual"""
        # En producción, esto obtendría el precio actual y calcularía la cantidad
        # respetando los lot sizes del exchange
        return size_usd / 1000  # Placeholder
    
    def _update_active_positions(self, symbol: str, action: str, order_result: Dict, signal: Dict):
        """Actualiza el registro de posiciones activas"""
        if symbol not in self.active_positions:
            self.active_positions[symbol] = []
        
        position = {
            'symbol': symbol,
            'action': action,
            'entry_price': float(order_result.get('fills', [{}])[0].get('price', 0)),
            'quantity': float(order_result.get('executedQty', 0)),
            'entry_time': datetime.now(),
            'signal': signal,
            'order_id': order_result.get('orderId')
        }
        
        self.active_positions[symbol].append(position)
    
    async def monitor_active_positions(self):
        """Monitorea posiciones activas y gestiona salidas"""
        for symbol, positions in self.active_positions.items():
            for position in positions[:]:  # Copia para modificar durante iteración
                exit_signal = await self._check_exit_conditions(symbol, position)
                
                if exit_signal:
                    await self._exit_position(symbol, position, exit_signal)
                    self.active_positions[symbol].remove(position)
    
    async def _check_exit_conditions(self, symbol: str, position: Dict) -> Optional[Dict]:
        """Verifica condiciones de salida para una posición"""
        current_time = datetime.now()
        holding_time = current_time - position['entry_time']
        
        # Salida por tiempo (especialmente para scalping)
        if (holding_time > timedelta(minutes=position['signal'].get('max_holding_time', 120)) and
            position['signal'].get('strategy') == 'scalping'):
            return {'reason': 'time_exit', 'type': 'TIME_STOP'}
        
        # Obtener precio actual
        try:
            ticker = await self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            entry_price = position['entry_price']
            
            # Calcular PnL
            if position['action'] == 'BUY':
                pnl_pct = (current_price - entry_price) / entry_price
                stop_loss = -position['signal'].get('stop_loss', 0.02)
                profit_target = position['signal'].get('profit_target', 0.03)
            else:  # SELL
                pnl_pct = (entry_price - current_price) / entry_price
                stop_loss = -position['signal'].get('stop_loss', 0.02)
                profit_target = position['signal'].get('profit_target', 0.03)
            
            # Verificar stop loss
            if pnl_pct <= stop_loss:
                return {'reason': 'stop_loss', 'type': 'STOP_LOSS', 'pnl': pnl_pct}
            
            # Verificar profit target
            if pnl_pct >= profit_target:
                return {'reason': 'profit_target', 'type': 'TAKE_PROFIT', 'pnl': pnl_pct}
                
        except Exception as e:
            self.logger.error(f"Error verificando condiciones de salida: {e}")
        
        return None
    
    async def _exit_position(self, symbol: str, position: Dict, exit_signal: Dict):
        """Ejecuta salida de posición"""
        try:
            if position['action'] == 'BUY':
                # Vender para cerrar posición larga
                order_result = await self.client.order_market_sell(
                    symbol=symbol,
                    quantity=position['quantity']
                )
            else:  # SELL
                # Comprar para cerrar posición corta
                order_result = await self.client.order_market_buy(
                    symbol=symbol,
                    quantity=position['quantity']
                )
            
            # Registrar resultado
            exit_price = float(order_result.get('fills', [{}])[0].get('price', 0))
            entry_price = position['entry_price']
            
            if position['action'] == 'BUY':
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
            
            trade_result = {
                'symbol': symbol,
                'entry_action': position['action'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': position['quantity'],
                'pnl_pct': pnl_pct,
                'exit_reason': exit_signal['reason'],
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'holding_time': datetime.now() - position['entry_time']
            }
            
            self.logger.info(f"Posición cerrada: {trade_result}")
            
            # Actualizar métricas de performance
            self._update_performance_metrics(trade_result)
            
        except Exception as e:
            self.logger.error(f"Error cerrando posición: {e}")
    
    def _update_performance_metrics(self, trade_result: Dict):
        """Actualiza métricas de performance"""
        symbol = trade_result['symbol']
        
        if symbol not in self.performance_tracker:
            self.performance_tracker[symbol] = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        metrics = self.performance_tracker[symbol]
        metrics['total_trades'] += 1
        metrics['total_pnl'] += trade_result['pnl_pct']
        
        if trade_result['pnl_pct'] > 0:
            metrics['winning_trades'] += 1
        else:
            metrics['losing_trades'] += 1
        
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
    
    def get_performance_report(self) -> Dict:
        """Genera reporte de performance"""
        return self.performance_tracker