# tests/test_entry_timing_multilayer.py
# ===============================================================
# 🧪 Test Mejorado del Entry Timing Analyzer Multilayer
# ---------------------------------------------------------------
# Incluye:
#   ✅ Datos de prueba realistas generados dinámicamente
#   ✅ Múltiples escenarios de prueba (long, short, pullback, rebote, contratendencia)
#   ✅ Datos técnicos consistentes (tendencia y MACD alineados)
#   ✅ Umbrales de scoring ajustados
#   ✅ Reporte detallado con métricas
#   ✅ Manejo robusto de errores y limpieza
# ===============================================================

import os
import sys
import json
import time
import tempfile
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.entry_timing_analyzer_multilayer import evaluar_timing_multilayer

console = Console()

# ===============================================================
# 🧩 Configuración de pruebas
# ===============================================================

class TestConfig:
    """Configuración global para pruebas"""
    TEMP_DIR = None
    TEST_FILES = []
    
    @classmethod
    def setup(cls):
        """Configura entorno de pruebas"""
        cls.TEMP_DIR = tempfile.mkdtemp(prefix="trading_test_")
        cls.TEST_FILES = []
        
    @classmethod
    def cleanup(cls):
        """Limpia archivos temporales"""
        if cls.TEMP_DIR and os.path.exists(cls.TEMP_DIR):
            shutil.rmtree(cls.TEMP_DIR, ignore_errors=True)

# ===============================================================
# 🎯 Generador de Datos de Prueba Realistas y Consistentes
# ===============================================================

def generate_realistic_data(direction="long", volatility=0.01, trend_strength=0.05):
    """
    Genera datos OHLCV realistas con MACD coherente garantizado
    """
    np.random.seed(42)
    
    n_candles = 100
    base_price = 50000
    time_index = pd.date_range(start="2025-01-01", periods=n_candles, freq="3min")
    
    # Generar tendencia fuerte con baja volatilidad
    if direction == "long":
        # Tendencia alcista muy fuerte para asegurar MACD positivo
        trend = np.linspace(0, 0.15, n_candles)  # 15% aumento
        # Añadir sesgo positivo adicional para MACD
        trend += np.linspace(0, 0.03, n_candles)
    else:  # short
        # Tendencia bajista muy fuerte para asegurar MACD negativo
        trend = np.linspace(0, -0.15, n_candles)  # 15% disminución
        # Añadir sesgo negativo adicional para MACD
        trend += np.linspace(0, -0.03, n_candles)
    
    # Muy baja volatilidad para reducir ruido
    volatility = 0.003
    noise = np.random.normal(0, volatility, n_candles)
    prices = base_price * (1 + trend + noise)
    
    # Crear OHLCV
    data = pd.DataFrame({
        'timestamp': time_index,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, volatility/2, n_candles))),
        'low': prices * (1 - np.abs(np.random.normal(0, volatility/2, n_candles))),
        'close': prices,
        'volume': np.random.randint(100, 1000, n_candles)
    })
    
    # Asegurar consistencia OHLC
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)
    
    # Garantizar MACD coherente ajustando los precios si es necesario
    if direction == "long":
        # Asegurar tendencia alcista adicional
        data['close'] = data['close'] * (1 + np.linspace(0, 0.005, n_candles))
    else:  # short
        # Asegurar tendencia bajista adicional
        data['close'] = data['close'] * (1 - np.linspace(0, 0.005, n_candles))
    
    return data

def create_pullback_data_con_consistent_macd():
    """Crea datos con pullback saludable y MACD coherente garantizado"""
    np.random.seed(123)
    
    n_candles = 100
    base_price = 50000
    time_index = pd.date_range(start="2025-01-01", periods=n_candles, freq="3min")
    
    # Tendencia alcista muy fuerte con pullback temporal
    trend = np.linspace(0, 0.15, n_candles)  # 15% total
    pullback_start = 40
    pullback_end = 60
    trend[pullback_start:pullback_end] = np.linspace(0.08, 0.10, pullback_end-pullback_start)
    
    # Añadir sesgo positivo para MACD
    trend += np.linspace(0, 0.03, n_candles)
    
    # Muy baja volatilidad
    volatility = 0.003
    noise = np.random.normal(0, volatility, n_candles)
    prices = base_price * (1 + trend + noise)
    
    data = pd.DataFrame({
        'timestamp': time_index,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, volatility/2, n_candles))),
        'low': prices * (1 - np.abs(np.random.normal(0, volatility/2, n_candles))),
        'close': prices,
        'volume': np.random.randint(100, 1000, n_candles)
    })
    
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)
    
    # Garantizar MACD positivo adicional
    data['close'] = data['close'] * (1 + np.linspace(0, 0.005, n_candles))
    
    return data

def create_bounce_data():
    """Crea datos con rebote técnico y MACD coherente garantizado"""
    np.random.seed(456)
    
    n_candles = 100
    base_price = 50000
    time_index = pd.date_range(start="2025-01-01", periods=n_candles, freq="3min")
    
    # Tendencia bajista muy fuerte con rebote técnico
    trend = np.linspace(0, -0.15, n_candles)  # 15% disminución
    bounce_start = 80
    trend[bounce_start:] = np.linspace(-0.10, -0.08, n_candles - bounce_start)
    
    # Añadir sesgo negativo para MACD
    trend += np.linspace(0, -0.03, n_candles)
    
    # Muy baja volatilidad
    volatility = 0.003
    noise = np.random.normal(0, volatility, n_candles)
    prices = base_price * (1 + trend + noise)
    
    data = pd.DataFrame({
        'timestamp': time_index,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, volatility/2, n_candles))),
        'low': prices * (1 - np.abs(np.random.normal(0, volatility/2, n_candles))),
        'close': prices,
        'volume': np.random.randint(100, 1000, n_candles)
    })
    
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)
    
    # Garantizar MACD negativo adicional
    data['close'] = data['close'] * (1 - np.linspace(0, 0.005, n_candles))
    
    return data

def create_counter_trend_data():
    """Crea datos con contratendencia y MACD coherente garantizado"""
    np.random.seed(789)
    
    n_candles = 100
    base_price = 50000
    time_index = pd.date_range(start="2025-01-01", periods=n_candles, freq="3min")
    
    # Tendencia alcista muy fuerte para operación short
    trend = np.linspace(0, 0.15, n_candles)  # 15% aumento
    # Añadir sesgo positivo para MACD
    trend += np.linspace(0, 0.03, n_candles)
    
    # Muy baja volatilidad
    volatility = 0.003
    noise = np.random.normal(0, volatility, n_candles)
    prices = base_price * (1 + trend + noise)
    
    data = pd.DataFrame({
        'timestamp': time_index,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, volatility/2, n_candles))),
        'low': prices * (1 - np.abs(np.random.normal(0, volatility/2, n_candles))),
        'close': prices,
        'volume': np.random.randint(100, 1000, n_candles)
    })
    
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)
    
    # Garantizar MACD positivo adicional
    data['close'] = data['close'] * (1 + np.linspace(0, 0.005, n_candles))
    
    return data


# ===============================================================
# 🧪 Casos de Prueba Mejorados
# ===============================================================

def run_test_case(test_name, data_3m, data_5m, data_15m, direction, expected_result=None):
    """Ejecuta un caso de prueba individual"""
    console.print(f"\n[bold cyan]▶️ Ejecutando: {test_name} ({direction.upper()})[/bold cyan]")
    
    try:
        # Ejecutar análisis
        start_time = time.time()
        result = evaluar_timing_multilayer(data_3m, data_5m, data_15m, direction)
        end_time = time.time()
        duration = end_time - start_time
        
        # Verificar resultado esperado
        success = True
        if expected_result is not None:
            if expected_result == "valid" and not result['valido']:
                console.print(f"❌ [red]Esperado: válido, obtenido: inválido[/red]")
                success = False
            elif expected_result == "invalid" and result['valido']:
                console.print(f"❌ [red]Esperado: inválido, obtenido: válido[/red]")
                success = False
        
        # Mostrar resultados
        if success:
            console.print(f"✅ [green]✓ Test '{test_name}' completado en {duration:.2f}s[/green]")
        else:
            console.print(f"❌ [red]✗ Test '{test_name}' falló[/red]")
        
        # Mostrar detalles clave
        table = Table(title=f"Detalles - {test_name}")
        table.add_column("Campo", style="cyan")
        table.add_column("Valor", style="magenta")
        
        table.add_row("Válido", str(result['valido']))
        table.add_row("Score", str(result['score']))
        table.add_row("Etapa", result['etapa'])
        table.add_row("Razón", result['razon'])
        table.add_row("Tendencia 3m", result['tendencia_3m'])
        table.add_row("Tendencia 5m", result['tendencia_5m'])
        table.add_row("Tendencia 15m", result['tendencia_15m'])
        table.add_row("RSI 3m", str(result['rsi_3m']))
        table.add_row("MACD 3m", str(result['macd_3m']))
        
        console.print(table)
        
        return {
            "name": test_name,
            "success": success,
            "duration": duration,
            "result": result
        }
        
    except Exception as e:
        console.print(f"❌ [red]Error en test '{test_name}': {str(e)}[/red]")
        return {
            "name": test_name,
            "success": False,
            "duration": 0,
            "error": str(e)
        }

def run_all_tests():
    """Ejecuta todos los casos de prueba"""
    # Configurar entorno
    TestConfig.setup()
    
    console.rule("[bold cyan]🧪 INICIO DE TESTS — Entry Timing Analyzer Multilayer Mejorado[/bold cyan]")
    
    # Generar datos para todos los timeframes
    console.print("\n[bold yellow]📊 Generando datos de prueba...[/bold yellow]")
    
    # Casos de prueba con datos técnicos consistentes
    test_cases = [
        {
            "name": "Long en Tendencia Alcista",
            "data_3m": generate_realistic_data("long", volatility=0.01, trend_strength=0.05),
            "data_5m": generate_realistic_data("long", volatility=0.01, trend_strength=0.05),
            "data_15m": generate_realistic_data("long", volatility=0.01, trend_strength=0.05),
            "direction": "long",
            "expected": "valid"
        },
        {
            "name": "Short en Tendencia Bajista",
            "data_3m": generate_realistic_data("short", volatility=0.01, trend_strength=0.05),
            "data_5m": generate_realistic_data("short", volatility=0.01, trend_strength=0.05),
            "data_15m": generate_realistic_data("short", volatility=0.01, trend_strength=0.05),
            "direction": "short",
            "expected": "valid"
        },
        {
            "name": "Long en Contratendencia",
            "data_3m": generate_realistic_data("long", volatility=0.01, trend_strength=0.05),
            "data_5m": generate_realistic_data("long", volatility=0.01, trend_strength=0.05),
            "data_15m": generate_realistic_data("long", volatility=0.01, trend_strength=0.05),
            "direction": "short",
            "expected": "invalid"
        },
        {
            "name": "Pullback Saludable",
            "data_3m": create_pullback_data_con_consistent_macd(),
            "data_5m": create_pullback_data_con_consistent_macd(),
            "data_15m": create_pullback_data_con_consistent_macd(),
            "direction": "long",
            "expected": "valid"
        },
        {
            "name": "Rebote Técnico",
            "data_3m": create_bounce_data(),
            "data_5m": create_bounce_data(),
            "data_15m": create_bounce_data(),
            "direction": "long",
            "expected": "valid"
        },
        {
            "name": "Contratendencia Riesgosa",
            "data_3m": generate_realistic_data("long", volatility=0.01, trend_strength=0.05),
            "data_5m": generate_realistic_data("long", volatility=0.01, trend_strength=0.05),
            "data_15m": generate_realistic_data("long", volatility=0.01, trend_strength=0.05),
            "direction": "short",
            "expected": "invalid"
        }
    ]
    
    console.print("✅ Datos generados exitosamente")
    
    # Ejecutar tests con progreso
    results = []
    total_time = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        overall_task = progress.add_task("Ejecutando tests...", total=len(test_cases))
        
        for test_case in test_cases:
            result = run_test_case(
                test_case["name"],
                test_case["data_3m"],
                test_case["data_5m"],
                test_case["data_15m"],
                test_case["direction"],
                test_case["expected"]
            )
            results.append(result)
            total_time += result["duration"]
            progress.update(overall_task, advance=1)
    
    # Generar reporte detallado
    generate_report(results, total_time)
    
    # Limpiar
    TestConfig.cleanup()

def generate_report(results, total_time):
    """Genera reporte detallado de resultados"""
    console.rule("[bold green]📊 RESUMEN DE TESTS[/bold green]")
    
    # Tabla de resultados
    table = Table(title="Resultados de los Tests")
    table.add_column("Test", style="cyan")
    table.add_column("Estado", style="green")
    table.add_column("Duración (s)", style="magenta")
    table.add_column("Score", style="yellow")
    
    passed = 0
    total_score = 0
    
    for result in results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        score = result["result"]["score"] if "result" in result else "N/A"
        table.add_row(result["name"], status, f"{result['duration']:.2f}", str(score))
        
        if result["success"]:
            passed += 1
        if "result" in result:
            total_score += result["result"]["score"]
    
    console.print(table)
    console.print(f"\n[bold green]✅ {passed}/{len(results)} tests pasados[/bold green]")
    console.print(f"[bold cyan]⏱️ Tiempo total: {total_time:.2f}s[/bold cyan]")
    
    if len(results) > 0:
        avg_score = total_score / len([r for r in results if "result" in r])
        console.print(f"[bold yellow]📈 Score promedio: {avg_score:.2f}[/bold yellow]")
    
    # Análisis detallado
    console.print("\n[bold magenta]🔍 Análisis Detallado:[/bold magenta]")
    
    # Agrupar por tipo de resultado
    valid_results = [r for r in results if r["success"] and "result" in r]
    invalid_results = [r for r in results if not r["success"] and "result" in r]
    
    if valid_results:
        console.print("\n[green]✅ Tests Válidos:[/green]")
        for r in valid_results:
            console.print(f"  - {r['name']}: Score {r['result']['score']:.2f} ({r['result']['etapa']})")
    
    if invalid_results:
        console.print("\n[red]❌ Tests Inválidos:[/red]")
        for r in invalid_results:
            console.print(f"  - {r['name']}: {r['result']['razon']}")

# ===============================================================
# 🚀 Ejecución principal
# ===============================================================

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Tests interrumpidos por el usuario[/yellow]")
        TestConfig.cleanup()
    except Exception as e:
        console.print(f"\n[red]❌ Error crítico: {str(e)}[/red]")
        TestConfig.cleanup()