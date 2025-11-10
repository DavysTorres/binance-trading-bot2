# main.py
import os
import time
import json
import asyncio
import pandas as pd
from datetime import datetime
from binance.client import Client
from binance import AsyncClient

# ===============================
# 📦 Importación de módulos existentes + nuevos
# ===============================
from modules.market_scanner_module import MarketScanner
from data.data_loader import DataLoader
from indicators.signal_analyzer import detectar_senal
from core.signal_buffer import SignalBuffer
from ai.advanced_signal_validator import validar_confluencia
from core.signal_reevaluator import reevaluar_senales
from trading.order_manager import OrderManager
from core.trade_monitor import monitorear_operaciones
from core.report_manager import generar_reporte_diario
from core.notification_manager import send_info, send_critical
from indicators.volatility_analyzer import analizar_volatilidad

# Nuevos módulos de estrategias y riesgo
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.scalping_strategy import ScalpingStrategy
from strategies.base_strategy import StrategyManager
from strategy_executor import StrategyExecutor
from risk_manager import RiskManager
from risk_config import RiskConfig

# ===============================
# ⚙️ Configuración dinámica
# ===============================
with open("config/settings.json") as f:
    config = json.load(f)

with open("config/strategies.json") as f:
    strategy_config = json.load(f)

with open("config/risk_config.json") as f:
    risk_config_data = json.load(f)

API_KEY = config.get("api_key")
API_SECRET = config.get("api_secret")
OHLCV_INTERVAL = config.get("ohlcv_interval", "3m")
OHLCV_LIMIT = config.get("ohlcv_limit", 150)
SCAN_INTERVAL = config.get("scan_interval_minutes", 10)
RETRY_INTERVAL = config.get("retry_interval_seconds", 60)

# ===============================
# 🏗️ Inicialización de componentes
# ===============================
class TradingBot:
    def __init__(self):
        # Clientes
        self.sync_client = Client(API_KEY, API_SECRET)
        self.async_client = None
        
        # DataLoader mejorado
        self.data_loader = DataLoader(self.sync_client)
        
        # Resto de componentes existentes
        self.scanner = MarketScanner()
        self.order_manager = OrderManager()
        self.buffer = SignalBuffer(config.get("signals_validated_path", "signals_validated.json"))
        
        # Nuevos componentes
        self.risk_manager = None
        self.strategy_manager = None
        self.executor = None
        
    async def initialize_async_components(self):
        """Inicializa componentes asíncronos"""
        try:
            self.async_client = await AsyncClient.create(API_KEY, API_SECRET)
            
            # Configurar DataLoader con cliente asíncrono
            self.data_loader.set_clients(self.sync_client, self.async_client)
            
            # Resto de inicialización...
            risk_config = RiskConfig()
            self.risk_manager = RiskManager(risk_config, self.async_client)
            
            self.strategy_manager = StrategyManager(strategy_config)
            
            # Registrar estrategias...
            self.strategy_manager.register_strategy(
                MomentumStrategy(strategy_config['momentum'])
            )
            # ... resto de estrategias
            
            self.executor = StrategyExecutor(
                self.strategy_manager, 
                self.risk_manager, 
                self.async_client
            )
            
            print("✅ Componentes asíncronos inicializados")
            
        except Exception as e:
            print(f"❌ Error inicializando componentes asíncronos: {e}")
            raise

    async def close_async_connections(self):
        """Cierra conexiones asíncronas"""
        if self.async_client:
            await self.async_client.close_connection()

# ===============================
# 🔍 ESCANEO DE MERCADO MEJORADO
# ===============================
async def ejecutar_escaneo_avanzado(bot):
    """
    Escaneo mejorado que combina detección tradicional + nuevas estrategias
    """
    print(f"\n🔁 Escaneando mercado completo — {datetime.now().strftime('%H:%M:%S')}")

    detectadas_tradicional = 0
    detectadas_estrategias = 0
    total = 0

    try:
        # Análisis de mercado global (existente)
        sesgo = bot.scanner.detectar_sesgo_mercado()
        intensidad = bot.scanner.calcular_intensidad_mercado()
        print(f"🧭 Sesgo global: {sesgo.upper()} | 🌡️ Intensidad: {intensidad:.2f}")

        top_symbols = bot.scanner.obtener_mejores_symbols()
        if not top_symbols:
            print("⚠️ No se encontraron símbolos destacados.")
            return 0, 0

        for symbol in top_symbols:
            print(f"\n🔍 Analizando {symbol}...")
            total += 1

            # Obtener datos para múltiples timeframes
            market_data = await obtener_datos_multitimeframe(bot, symbol)
            if not market_data:
                continue

            # 1️⃣ DETECCIÓN TRADICIONAL (existente)
            df_principal = market_data.get('15m')  # Usar 15m para detección tradicional
            if df_principal is not None and not df_principal.empty:
                # Filtrar por volatilidad
                if analizar_volatilidad(df_principal):
                    # Detección técnica tradicional
                    resultado = detectar_senal(df_principal)
                    if resultado.get("entrada"):
                        print(f"✅ Señal tradicional en {symbol} ({resultado['estrategia']})")
                        bot.buffer.add(symbol, resultado["tipo"], resultado["estrategia"])
                        detectadas_tradicional += 1

            # 2️⃣ NUEVAS ESTRATEGIAS + GESTIÓN DE RIESGO
            try:
                # Ejecutar ciclo de estrategias avanzadas
                result = await bot.executor.execute_strategy_cycle(symbol, market_data)
                
                if result['executed']:
                    print(f"🚀 Operación ejecutada con estrategias avanzadas: {symbol}")
                    detectadas_estrategias += 1
                elif result.get('reason') == 'risk_rejection':
                    print(f"🛑 Operación rechazada por riesgo: {symbol}")
                else:
                    print(f"⏸️ Sin señal de estrategias avanzadas: {symbol}")
                    
            except Exception as e:
                print(f"⚠️ Error en estrategias avanzadas para {symbol}: {e}")

            print("-" * 45)
            await asyncio.sleep(1.2)  # Sleep asíncrono

    except Exception as e:
        print(f"❌ Error en escaneo avanzado: {e}")
        send_critical("Sistema", f"Error durante escaneo avanzado: {e}")

    print(f"\n📊 Resumen escaneo:")
    print(f"   • Analizadas: {total}")
    print(f"   • Señales tradicionales: {detectadas_tradicional}")
    print(f"   • Señales avanzadas: {detectadas_estrategias}")
    
    return detectadas_tradicional, detectadas_estrategias

async def obtener_datos_multitimeframe(bot, symbol: str):
    """Obtiene datos para múltiples timeframes (compatible con tus datos existentes)"""
    timeframes = ['1m', '5m', '15m', '1h', '4h']
    market_data = {}
    
    for tf in timeframes:
        try:
            # Usar tu data_loader existente para timeframes principales
            if tf in ['15m', '1h', '4h']:
                df = bot.data_loader.obtener_ohlcv(symbol, interval=tf, limit=100)
                if df is not None and not df.empty:
                    # Convertir a formato compatible con nuevas estrategias
                    df_compatible = df.rename(columns={
                        'Open': 'open', 'High': 'high', 'Low': 'low', 
                        'Close': 'close', 'Volume': 'volume'
                    })
                    market_data[tf] = df_compatible
            else:
                # Para timeframes más cortos, usar API asíncrona
                klines = await bot.async_client.get_historical_klines(
                    symbol, tf, "100 candle ago UTC"
                )
                
                if klines:
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convertir tipos de datos
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    market_data[tf] = df
                    
        except Exception as e:
            print(f"⚠️ Error obteniendo datos {tf} para {symbol}: {e}")
    
    return market_data

# ===============================
# 🔄 CICLO OPERATIVO PRINCIPAL MEJORADO
# ===============================
async def ciclo_principal_avanzado():
    print(f"\n🕒 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔄 Iniciando ciclo operativo avanzado...")

    # Inicializar bot
    bot = TradingBot()
    await bot.initialize_async_components()

    try:
        # 1️⃣ Escaneo técnico mejorado (tradicional + nuevas estrategias)
        trad_detectadas, avanzadas_detectadas = await ejecutar_escaneo_avanzado(bot)

        # 2️⃣ Reevaluación + Validación Avanzada (existente)
        print("\n🤖 Reevaluando y validando confluencias...")
        reevaluar_senales()

        # 3️⃣ Monitoreo de operaciones activas (mejorado)
        print("\n📈 Monitoreando operaciones...")
        await bot.executor.monitor_active_positions()  # Nuevo monitoreo
        monitorear_operaciones()  # Monitoreo existente

        # 4️⃣ Reporte diario mejorado
        print("\n🧾 Generando reporte diario avanzado...")
        generar_reporte_diario()
        
        # Reporte de performance de nuevas estrategias
        performance_report = bot.executor.get_performance_report()
        print("📊 Performance estrategias avanzadas:", performance_report)

        # 5️⃣ Notificación del ciclo mejorada
        try:
            resumen = (
                f"📊 *Resumen del Ciclo Avanzado*\n"
                f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"🔍 Señales tradicionales: {trad_detectadas}\n"
                f"🚀 Señales avanzadas: {avanzadas_detectadas}\n"
                f"📈 Operaciones activas: {len(bot.executor.active_positions)}\n"
            )
            send_info("Sistema", resumen)
        except Exception as e:
            print(f"⚠️ Error enviando resumen del ciclo: {e}")

        print("\n✅ Ciclo operativo avanzado finalizado.\n")

    finally:
        # Limpiar recursos
        await bot.close_async_connections()

# ===============================
# 🔁 LOOP CONTINUO MEJORADO
# ===============================
async def main_async():
    print("🚀 Iniciando bot de trading avanzado con gestión de riesgo...\n")

    while True:
        try:
            await ciclo_principal_avanzado()
            print(f"⏳ Esperando {SCAN_INTERVAL} minutos antes del próximo ciclo...\n")
            await asyncio.sleep(SCAN_INTERVAL * 60)

        except KeyboardInterrupt:
            print("\n🛑 Ejecución detenida manualmente.")
            break

        except Exception as e:
            print(f"⚠️ Error crítico en el ciclo principal: {e}")
            send_critical("Sistema", f"Error crítico en el ciclo principal: {e}")
            print(f"⏳ Reintentando en {RETRY_INTERVAL} segundos...\n")
            await asyncio.sleep(RETRY_INTERVAL)

if __name__ == "__main__":
    # Ejecutar versión asíncrona
    asyncio.run(main_async())