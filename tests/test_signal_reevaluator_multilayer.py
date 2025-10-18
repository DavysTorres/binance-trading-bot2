

# tests/test_signal_reevaluator_improved.py
# ===============================================================
# 🧪 Test Mejorado del Reevaluador Multilayer con Estado
# ---------------------------------------------------------------
# Simula señales y datos de mercado realistas para probar:
#   ✅ Reevaluación multilayer con estado persistente
#   ✅ Manejo de diferentes regímenes de mercado
#   ✅ Gestión de riesgo dinámica
#   ✅ Manejo de pullbacks y confirmaciones
#   ✅ Bloqueo temporal ante errores consecutivos
#   ✅ Reportes detallados de resultados
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
from unittest.mock import patch, MagicMock
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.signal_reevaluator import reevaluar_senales, ReevaluadorEstado
from core.signal_buffer import SignalBuffer
from data.data_loader import DataLoader
from ai.entry_timing_analyzer_multilayer import evaluar_timing_multilayer

console = Console()

# ===============================================================
# 🧩 Configuración de pruebas mejorada
# ===============================================================

class TestConfig:
    """Configuración global para pruebas con mock de dependencias"""
    TEMP_DIR = None
    TEST_FILES = []
    MOCK_MARKET_DATA = {}
    
    @classmethod
    def setup(cls):
        """Configura entorno de pruebas con mock de dependencias"""
        cls.TEMP_DIR = tempfile.mkdtemp(prefix="trading_test_")
        cls.TEST_FILES = []
        
        # Mock de Binance client
        mock_client = MagicMock()
        mock_client.get_symbol_info.return_value = {
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                {"filterType": "MIN_NOTIONAL", "minNotional": "10.0"}
            ]
        }
        
        # Mock de DataLoader para devolver datos de prueba
        mock_data_loader = MagicMock()
        mock_data_loader.obtener_ohlcv = cls.mock_obtener_ohlcv
        
        # Patch de dependencias
        patch('core.signal_reevaluator.client', mock_client).start()
        patch('core.signal_reevaluator.data_loader', mock_data_loader).start()
        patch('core.signal_reevaluator.order_manager', MagicMock()).start()
        patch('core.signal_reevaluator.risk_manager', MagicMock()).start()
        
    @classmethod
    def mock_obtener_ohlcv(cls, symbol, interval, limit):
        """Mock de obtención de datos OHLCV"""
        if symbol not in cls.MOCK_MARKET_DATA:
            cls.MOCK_MARKET_DATA[symbol] = cls.generate_market_data(interval, limit)
        return cls.MOCK_MARKET_DATA[symbol]
    
    @classmethod
    def generate_market_data(cls, interval, limit):
        """Genera datos de mercado realistas"""
        np.random.seed(42)
        n_candles = limit
        base_price = 50000
        time_index = pd.date_range(
            start=datetime.now() - timedelta(minutes=n_candles*3), 
            periods=n_candles, 
            freq="3min"
        )
        
        # Tendencia general con variaciones
        if interval == "3m":
            trend = np.linspace(0, 0.05, n_candles)
        elif interval == "5m":
            trend = np.linspace(0, 0.04, n_candles)
        else:  # 15m
            trend = np.linspace(0, 0.06, n_candles)
        
        noise = np.random.normal(0, 0.01, n_candles)
        prices = base_price * (1 + trend + noise)
        
        data = pd.DataFrame({
            'timestamp': time_index,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_candles))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_candles))),
            'close': prices,
            'volume': np.random.randint(100, 1000, n_candles)
        })
        
        # Asegurar consistencia OHLC
        data['high'] = data[['high', 'open', 'close']].max(axis=1)
        data['low'] = data[['low', 'open', 'close']].min(axis=1)
        
        return data
    
    @classmethod
    def cleanup(cls):
        """Limpia archivos temporales y detiene patches"""
        if cls.TEMP_DIR and os.path.exists(cls.TEMP_DIR):
            shutil.rmtree(cls.TEMP_DIR, ignore_errors=True)
        
        # Detener patches
        patch.stopall()

# ===============================================================
# 🧪 Generadores de Escenarios de Prueba
# ===============================================================

def create_test_signal(symbol, tipo, score, estado="pendiente", 
                      estrategia="multilayer_test", datos_tecnicos=None):
    """Crea señal de prueba con parámetros configurables"""
    return {
        "symbol": symbol,
        "tipo": tipo,
        "estrategia": estrategia,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "score_total": score,
        "estado": estado,
        "intentos": 0,
        "datos_tecnicos": datos_tecnicos or {
            "rsi": 45.0,
            "macd": 0.002,
            "tendencia": "bullish"
        }
    }

def create_pullback_scenario():
    """Crea escenario con pullback saludable"""
    # Generar datos con pullback en el medio
    data = TestConfig.generate_market_data("3m", 150)
    
    # Modificar para crear pullback
    pullback_start = 70
    pullback_end = 90
    pullback_factor = 0.98  # 2% de caída
    
    data.loc[pullback_start:pullback_end, 'close'] *= pullback_factor
    data.loc[pullback_start:pullback_end, 'open'] *= pullback_factor
    data.loc[pullback_start:pullback_end, 'high'] *= pullback_factor
    data.loc[pullback_start:pullback_end, 'low'] *= pullback_factor
    
    # Actualizar high/low para consistencia
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)
    
    return data

def create_bounce_scenario():
    """Crea escenario con rebote técnico"""
    # Generar datos en tendencia bajista con rebote al final
    data = TestConfig.generate_market_data("3m", 150)
    
    # Modificar para crear tendencia bajista con rebote
    trend_factor = 0.995  # 0.5% de caída por vela
    data['close'] = data['close'] * (trend_factor ** np.arange(len(data)))
    
    # Crear rebote en las últimas velas
    bounce_start = 130
    bounce_factor = 1.02  # 2% de subida
    data.loc[bounce_start:, 'close'] *= bounce_factor
    data.loc[bounce_start:, 'open'] *= bounce_factor
    data.loc[bounce_start:, 'high'] *= bounce_factor
    data.loc[bounce_start:, 'low'] *= bounce_factor
    
    # Actualizar high/low para consistencia
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)
    
    return data

def create_counter_trend_scenario():
    """Crea escenario con contratendencia"""
    # Generar datos en tendencia alcista fuerte
    data = TestConfig.generate_market_data("3m", 150)
    
    # Modificar para crear tendencia alcista fuerte
    trend_factor = 1.008  # 0.8% de subida por vela
    data['close'] = data['close'] * (trend_factor ** np.arange(len(data)))
    
    # Actualizar high/low para consistencia
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)
    
    return data

# ===============================================================
# 🧪 Casos de Prueba Mejorados
# ===============================================================

def test_reevaluador_estado():
    """Prueba el manejo de estado del reevaluador"""
    console.print("\n[bold cyan]▶️ Prueba: Manejo de Estado[/bold cyan]")
    
    estado = ReevaluadorEstado()
    
    # Probar inicialización
    assert estado.registro_mercado['regime_actual'] == 'neutral'
    assert estado.registro_mercado['consecutivos_invalidos'] == 0
    
    # Probar actualización de régimen
    estado.actualizar_regimen('bullish_strong')
    assert estado.registro_mercado['regime_actual'] == 'bullish_strong'
    
    # Prober registro de evaluaciones
    resultado = {'valido': True, 'score': 0.8}
    estado.registrar_evaluacion(resultado)
    assert estado.registro_mercado['ultima_evaluacion'] == resultado
    assert estado.registro_mercado['consecutivos_invalidos'] == 0
    
    # Prober manejo de inválidos consecutivos
    estado.registrar_evaluacion({'valido': False, 'score': 0.3})
    assert estado.registro_mercado['consecutivos_invalidos'] == 1
    
    # Prober bloqueo temporal
    estado.bloquear_temporalmente(1)
    assert estado.registro_mercado['bloqueado_husto'] is not None
    assert not estado.debe_ejecutar()
    
    console.print("✅ [green]✓ Test de estado completado[/green]")
    return True

def test_reevaluador_confianza():
    """Prueba el cálculo de confianza adicional"""
    console.print("\n[bold cyan]▶️ Prueba: Cálculo de Confianza[/bold cyan]")
    
    estado = ReevaluadorEstado()
    
    # Probar confianza base
    confianza = estado.obtener_confianza_adicional()
    assert confianza == 1.0
    
    # Prober reducción por inválidos consecutivos
    estado.registrar_evaluacion({'valido': False, 'score': 0.3})
    estado.registrar_evaluacion({'valido': False, 'score': 0.2})
    confianza = estado.obtener_confianza_adicional()
    assert confianza < 1.0
    
    # Prober aumento por régimen fuerte
    estado.actualizar_regimen('bullish_strong')
    confianza = estado.obtener_confianza_adicional()
    assert confianza > 1.0
    
    console.print("✅ [green]✓ Test de confianza completado[/green]")
    return True

def test_reevaluador_escenarios():
    """Prueba diferentes escenarios de mercado"""
    console.print("\n[bold cyan]▶️ Prueba: Escenarios de Mercado[/bold cyan]")
    
    # Crear archivo temporal
    test_file = TestConfig.get_test_path("test_escenarios.json")
    TestConfig.TEST_FILES.append(test_file)
    
    try:
        # Crear señales para diferentes escenarios
        señales = [
            create_test_signal("BTCUSDT", "long", 0.85),  # Alta confianza
            create_test_signal("ETHUSDT", "short", 0.90),  # Alta confianza
            create_test_signal("ADAUSDT", "long", 0.25),  # Baja confianza
            create_test_signal("SOLUSDT", "short", 0.75, datos_tecnicos={"tendencia": "bullish"}),  # Contratendencia
        ]
        
        # Guardar señales
        with open(test_file, "w") as f:
            json.dump(señales, f, indent=4)
        
        # Crear buffer y reemplazar en reevaluador
        buffer = SignalBuffer(test_file)
        import core.signal_reevaluator as reevaluator
        reevaluator.buffer = buffer
        
        # Ejecutar reevaluación
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Reevaluando escenarios...", total=1)
            start_time = time.time()
            reevaluar_senales()
            progress.update(task, advance=1)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verificar resultados
        with open(test_file, "r") as f:
            result_data = json.load(f)
        
        # Analizar resultados
        resultados_analisis = []
        for signal in result_data:
            resultados_analisis.append({
                'symbol': signal['symbol'],
                'tipo': signal['tipo'],
                'estado_final': signal['estado'],
                'score': signal.get('score_total', 0)
            })
        
        # Mostrar resultados
        table = Table(title="Resultados de Escenarios")
        table.add_column("Símbolo", style="cyan")
        table.add_column("Tipo", style="magenta")
        table.add_column("Estado Final", style="green")
        table.add_column("Score", style="yellow")
        
        for resultado in resultados_analisis:
            table.add_row(
                resultado['symbol'],
                resultado['tipo'],
                resultado['estado_final'],
                str(resultado['score'])
            )
        
        console.print(table)
        console.print(f"✅ [green]✓ Test de escenarios completado en {duration:.2f}s[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"❌ [red]Error en test de escenarios: {str(e)}[/red]")
        return False

def test_reevaluador_pullback():
    """Prueba el manejo de pullbacks"""
    console.print("\n[bold cyan]▶️ Prueba: Manejo de Pullbacks[/bold cyan]")
    
    # Crear datos con pullback
    data_pullback = create_pullback_scenario()
    
    # Simular evaluación multilayer con pullback
    result = evaluar_timing_multilayer(data_pullback, data_pullback, data_pullback, "long")
    
    # Verificar que detecta pullback
    assert result.get('pullback_saludable', False) == True
    assert result['etapa'] == 'pullback'
    
    console.print("✅ [green]✓ Test de pullback completado[/green]")
    return True

def test_reevaluador_bounce():
    """Prueba el manejo de rebotes técnicos"""
    console.print("\n[bold cyan]▶️ Prueba: Manejo de Rebotes Técnicos[/bold cyan]")
    
    # Crear datos con rebote
    data_bounce = create_bounce_scenario()
    
    # Simular evaluación multilayer con rebote
    result = evaluar_timing_multilayer(data_bounce, data_bounce, data_bounce, "long")
    
    # Verificar que detecta rebote
    assert result.get('modo_rebote', False) == True
    assert result['etapa'] == 'rebote'
    
    console.print("✅ [green]✓ Test de rebote completado[/green]")
    return True

def test_reevaluador_counter_trend():
    """Prueba el manejo de contratendencia"""
    console.print("\n[bold cyan]▶️ Prueba: Manejo de Contratendencia[/bold cyan]")
    
    # Crear datos con contratendencia
    data_counter = create_counter_trend_scenario()
    
    # Simular evaluación multilayer con contratendencia
    result = evaluar_timing_multilayer(data_counter, data_counter, data_counter, "short")
    
    # Verifica que detecta contratendencia
    assert result['etapa'] == 'contratendencia'
    assert result['valido'] == False
    
    console.print("✅ [green]✓ Test de contratendencia completado[/green]")
    return True

# ===============================================================
# 🧪 Ejecución de Tests
# ===============================================================

def run_all_tests():
    """Ejecuta todos los casos de prueba"""
    # Configurar entorno
    TestConfig.setup()
    
    console.rule("[bold cyan]🧪 INICIO DE TESTS — Reevaluador Multilayer Mejorado[/bold cyan]")
    
    # Lista de tests
    tests = [
        ("Manejo de Estado", test_reevaluador_estado),
        ("Cálculo de Confianza", test_reevaluador_confianza),
        ("Escenarios de Mercado", test_reevaluador_escenarios),
        ("Pullback Saludable", test_reevaluador_pullback),
        ("Rebote Técnico", test_reevaluador_bounce),
        ("Contratendencia", test_reevaluador_counter_trend),
    ]
    
    # Ejecutar tests
    results = []
    total_time = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        overall_task = progress.add_task("Ejecutando tests...", total=len(tests))
        
        for test_name, test_func in tests:
            start_time = time.time()
            success = test_func()
            end_time = time.time()
            duration = end_time - start_time
            
            results.append({
                "name": test_name,
                "success": success,
                "duration": duration
            })
            total_time += duration
            progress.update(overall_task, advance=1)
    
    # Mostrar resumen
    console.rule("[bold green]📊 RESUMEN DE TESTS[/bold green]")
    
    # Tabla de resultados
    table = Table(title="Resultados de los Tests")
    table.add_column("Test", style="cyan")
    table.add_column("Estado", style="green")
    table.add_column("Duración (s)", style="magenta")
    
    passed = 0
    for result in results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        table.add_row(result["name"], status, f"{result['duration']:.2f}")
        if result["success"]:
            passed += 1
    
    console.print(table)
    console.print(f"\n[bold green]✅ {passed}/{len(tests)} tests pasados[/bold green]")
    console.print(f"[bold cyan]⏱️ Tiempo total: {total_time:.2f}s[/bold cyan]")
    
    # Mostrar detalles de fallos
    failed_tests = [r for r in results if not r["success"]]
    if failed_tests:
        console.print("\n[bold red]❌ Tests Fallidos:[/bold red]")
        for test in failed_tests:
            console.print(f"  - {test['name']}")
    
    # Limpiar
    TestConfig.cleanup()

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