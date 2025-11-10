# portfolio_manager.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from risk.risk_config import RiskConfig, RiskLevel

class PortfolioManager:
    """Gestiona exposición y diversificación del portafolio"""
    
    def __init__(self, config: RiskConfig, binance_client, logger):
        self.config = config
        self.client = binance_client
        self.logger = logger
        self.correlation_matrix = {}
        
    async def analyze_portfolio_risk(self, symbol: str, signal_type: str) -> Dict[str, Any]:
        """Analiza riesgo de portafolio para una nueva operación"""
        risk_score = 0
        recommendations = []
        
        try:
            # Obtener posiciones actuales
            current_positions = await self._get_current_positions()
            total_portfolio_value = await self._get_total_portfolio_value()
            
            # 1. Exposición por símbolo
            symbol_exposure = await self._analyze_symbol_exposure(symbol, current_positions, total_portfolio_value)
            risk_score += symbol_exposure['score']
            if symbol_exposure['level'] == RiskLevel.HIGH:
                recommendations.append(symbol_exposure['message'])
            
            # 2. Exposición correlacionada
            correlation_risk = await self._analyze_correlation_exposure(symbol, current_positions, total_portfolio_value)
            risk_score += correlation_risk['score']
            if correlation_risk['level'] == RiskLevel.HIGH:
                recommendations.append(correlation_risk['message'])
            
            # 3. Exposición por sector
            sector_risk = await self._analyze_sector_exposure(symbol, current_positions)
            risk_score += sector_risk['score']
            if sector_risk['level'] == RiskLevel.HIGH:
                recommendations.append(sector_risk['message'])
            
            # 4. Concentración general
            concentration_risk = await self._analyze_portfolio_concentration(current_positions, total_portfolio_value)
            risk_score += concentration_risk['score']
            if concentration_risk['level'] == RiskLevel.HIGH:
                recommendations.append(concentration_risk['message'])
            
            approved = risk_score <= 4  # Máximo score para aprobación en portafolio
            
            return {
                'approved': approved,
                'score': risk_score,
                'recommendations': recommendations,
                'total_exposure': total_portfolio_value,
                'current_positions': len(current_positions),
                'symbol_exposure': symbol_exposure,
                'correlation_exposure': correlation_risk,
                'sector_exposure': sector_risk,
                'concentration': concentration_risk,
                'confidence': 1 - (risk_score / 10)  # 0-1 confidence score
            }
            
        except Exception as e:
            self.logger.error(f"Error en análisis de portafolio: {e}")
            return {
                'approved': False,
                'score': 10,
                'recommendations': ['Error en análisis de portafolio'],
                'error': str(e)
            }
    
    async def _analyze_symbol_exposure(self, symbol: str, positions: Dict, total_value: float) -> Dict[str, Any]:
        """Analiza exposición específica al símbolo"""
        current_exposure = positions.get(symbol, 0)
        exposure_ratio = current_exposure / total_value if total_value > 0 else 0
        
        max_exposure = self.config.MAX_EXPOSURE_PER_SYMBOL
        
        if exposure_ratio > max_exposure * 0.9:
            return {
                'level': RiskLevel.HIGH,
                'score': 4,
                'exposure_ratio': exposure_ratio,
                'message': f"Exposición a {symbol} muy alta: {exposure_ratio:.2%}"
            }
        elif exposure_ratio > max_exposure * 0.7:
            return {
                'level': RiskLevel.MEDIUM,
                'score': 2,
                'exposure_ratio': exposure_ratio,
                'message': f"Exposición a {symbol} elevada: {exposure_ratio:.2%}"
            }
        else:
            return {
                'level': RiskLevel.LOW,
                'score': 0,
                'exposure_ratio': exposure_ratio,
                'message': f"Exposición a {symbol} dentro de límites"
            }
    
    async def _analyze_correlation_exposure(self, symbol: str, positions: Dict, total_value: float) -> Dict[str, Any]:
        """Analiza exposición a activos correlacionados"""
        correlated_symbols = await self._get_correlated_assets(symbol)
        correlated_exposure = 0
        
        for correlated_symbol in correlated_symbols:
            if correlated_symbol in positions:
                correlated_exposure += positions[correlated_symbol]
        
        total_correlated_ratio = correlated_exposure / total_value if total_value > 0 else 0
        
        if total_correlated_ratio > self.config.MAX_CORRELATED_EXPOSURE * 0.8:
            return {
                'level': RiskLevel.HIGH,
                'score': 3,
                'correlated_ratio': total_correlated_ratio,
                'message': f"Exposición correlacionada alta: {total_correlated_ratio:.2%}"
            }
        elif total_correlated_ratio > self.config.MAX_CORRELATED_EXPOSURE * 0.5:
            return {
                'level': RiskLevel.MEDIUM,
                'score': 1,
                'correlated_ratio': total_correlated_ratio,
                'message': f"Exposición correlacionada media: {total_correlated_ratio:.2%}"
            }
        else:
            return {
                'level': RiskLevel.LOW,
                'score': 0,
                'correlated_ratio': total_correlated_ratio,
                'message': "Exposición correlacionada dentro de límites"
            }
    
    async def _analyze_sector_exposure(self, symbol: str, positions: Dict) -> Dict[str, Any]:
        """Analiza exposición por sector/mercado"""
        # Agrupar símbolos por categoría (ej: BTC, ETH = Large Cap, otros = Altcoins)
        sector_exposure = await self._calculate_sector_exposure(symbol, positions)
        
        max_sector_exposure = 0.4  # 40% máximo por sector
        
        if sector_exposure > max_sector_exposure:
            return {
                'level': RiskLevel.HIGH,
                'score': 3,
                'sector_exposure': sector_exposure,
                'message': f"Exposición sectorial alta: {sector_exposure:.2%}"
            }
        else:
            return {
                'level': RiskLevel.LOW,
                'score': 0,
                'sector_exposure': sector_exposure,
                'message': "Exposición sectorial dentro de límites"
            }
    
    async def _analyze_portfolio_concentration(self, positions: Dict, total_value: float) -> Dict[str, Any]:
        """Analiza concentración general del portafolio"""
        if not positions:
            return {'level': RiskLevel.LOW, 'score': 0, 'concentration': 0}
        
        # Calcular índice Gini de concentración
        exposures = list(positions.values())
        if total_value > 0:
            exposures = [x / total_value for x in exposures]
        
        sorted_exposures = sorted(exposures)
        n = len(sorted_exposures)
        gini = 1 - 2 * sum((i + 1) * x for i, x in enumerate(sorted_exposures)) / (n * sum(sorted_exposures)) if sum(sorted_exposures) > 0 else 0
        
        if gini > 0.6:  # Alta concentración
            return {
                'level': RiskLevel.HIGH,
                'score': 2,
                'concentration': gini,
                'message': f"Portafolio muy concentrado (Gini: {gini:.3f})"
            }
        elif gini > 0.4:
            return {
                'level': RiskLevel.MEDIUM,
                'score': 1,
                'concentration': gini,
                'message': f"Portafolio moderadamente concentrado (Gini: {gini:.3f})"
            }
        else:
            return {
                'level': RiskLevel.LOW,
                'score': 0,
                'concentration': gini,
                'message': f"Diversificación adecuada (Gini: {gini:.3f})"
            }
    
    async def _get_correlated_assets(self, symbol: str, threshold: float = 0.7) -> List[str]:
        """Obtiene activos correlacionados con el símbolo"""
        # Implementar lógica de correlación basada en datos históricos
        # Por ahora, retornar grupos predefinidos
        correlation_groups = {
            'BTCUSDT': ['ETHUSDT', 'LTCUSDT', 'BCHUSDT'],
            'ETHUSDT': ['BTCUSDT', 'LINKUSDT', 'ADAUSDT'],
            'ADAUSDT': ['DOTUSDT', 'SOLUSDT', 'AVAXUSDT']
        }
        
        return correlation_groups.get(symbol, [])
    
    async def _get_current_positions(self) -> Dict[str, float]:
        """Obtiene posiciones actuales desde Binance"""
        try:
            # Para spot
            account = await self.client.get_account()
            positions = {}
            
            for balance in account['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                if total > 0 and asset != 'USDT':
                    symbol = f"{asset}USDT"
                    # Obtener precio actual para calcular valor en USDT
                    ticker = await self.client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker['price'])
                    positions[symbol] = total * price
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error obteniendo posiciones: {e}")
            return {}
    
    async def _get_total_portfolio_value(self) -> float:
        """Calcula valor total del portafolio"""
        positions = await self._get_current_positions()
        return sum(positions.values())
    
    async def _calculate_sector_exposure(self, symbol: str, positions: Dict) -> float:
        """Calcula exposición por sector"""
        # Lógica simplificada de categorización
        large_caps = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        defi_tokens = ['UNIUSDT', 'AAVEUSDT', 'COMPUSDT']
        
        symbol_sector = 'large_cap' if symbol in large_caps else 'defi' if symbol in defi_tokens else 'altcoin'
        
        sector_value = 0
        for pos_symbol, value in positions.items():
            if (symbol_sector == 'large_cap' and pos_symbol in large_caps) or \
               (symbol_sector == 'defi' and pos_symbol in defi_tokens) or \
               (symbol_sector == 'altcoin' and pos_symbol not in large_caps + defi_tokens):
                sector_value += value
        
        total_value = await self._get_total_portfolio_value()
        return sector_value / total_value if total_value > 0 else 0