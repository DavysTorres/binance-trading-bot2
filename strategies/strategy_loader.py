# strategies/strategy_loader.py
"""
Loader auxiliar para importar estrategias correctamente
"""
import importlib
from typing import Dict, Any

def load_strategy(strategy_name: str, config: Dict[str, Any]):
    """
    Carga una estrategia dinámicamente
    """
    strategy_mapping = {
        'momentum': ('momentum_strategy', 'MomentumStrategy'),
        'mean_reversion': ('mean_reversion_strategy', 'MeanReversionStrategy'),
        'breakout': ('breakout_strategy', 'BreakoutStrategy'), 
        'scalping': ('scalping_strategy', 'ScalpingStrategy')
    }
    
    if strategy_name not in strategy_mapping:
        raise ValueError(f"Estrategia no soportada: {strategy_name}")
    
    module_name, class_name = strategy_mapping[strategy_name]
    
    try:
        module = importlib.import_module(f'strategies.{module_name}')
        strategy_class = getattr(module, class_name)
        return strategy_class(config)
    except Exception as e:
        raise ImportError(f"No se pudo cargar {strategy_name}: {e}")

# Importaciones directas para compatibilidad
try:
    from strategies.momentum_strategy import MomentumStrategy
except ImportError:
    MomentumStrategy = None

try:
    from strategies.mean_reversion_strategy import MeanReversionStrategy
except ImportError:
    MeanReversionStrategy = None

try:
    from strategies.breakout_strategy import BreakoutStrategy
except ImportError:
    BreakoutStrategy = None

try:
    from strategies.scalping_strategy import ScalpingStrategy  
except ImportError:
    ScalpingStrategy = None
