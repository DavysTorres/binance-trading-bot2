# fix_strategies_imports.py
import os
import importlib
import sys

def fix_strategy_imports():
    """Repara los imports de las estrategias"""
    print("🔧 REPARANDO IMPORTS DE ESTRATEGIAS")
    print("=" * 50)
    
    # Verificar que los archivos existen
    strategy_files = [
        'strategies/momentum_strategy.py',
        'strategies/mean_reversion_strategy.py', 
        'strategies/breakout_strategy.py',
        'strategies/scalping_strategy.py'
    ]
    
    for file in strategy_files:
        if os.path.exists(file):
            print(f"✅ {file} - Existe")
        else:
            print(f"❌ {file} - No existe")
            return False
    
    # Probar imports directos
    strategies_to_test = [
        'momentum_strategy',
        'mean_reversion_strategy',
        'breakout_strategy', 
        'scalping_strategy'
    ]
    
    for strategy in strategies_to_test:
        try:
            module = importlib.import_module(f'strategies.{strategy}')
            print(f"✅ strategies.{strategy} - Import OK")
            
            # Verificar que la clase existe
            class_name = strategy.replace('_strategy', '').title().replace('_', '') + 'Strategy'
            if hasattr(module, class_name):
                print(f"✅ {class_name} - Clase encontrada")
            else:
                print(f"❌ {class_name} - Clase NO encontrada en {strategy}")
                
        except Exception as e:
            print(f"❌ strategies.{strategy} - Error: {e}")
    
    return True

def create_strategy_loader():
    """Crea un loader auxiliar para estrategias"""
    loader_content = '''# strategies/strategy_loader.py
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
'''
    
    with open('strategies/strategy_loader.py', 'w', encoding='utf-8') as f:
        f.write(loader_content)
    print("✅ strategy_loader.py creado")

if __name__ == "__main__":
    fix_strategy_imports()
    create_strategy_loader()