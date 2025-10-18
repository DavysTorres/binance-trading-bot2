# main.py
import os
import time
import json
from datetime import datetime
from binance.client import Client

# ===============================
# 📦 Importación de módulos
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

# ===============================
# ⚙️ Configuración dinámica
# ===============================
with open("config/settings.json") as f:
    config = json.load(f)

API_KEY = config.get("api_key")
API_SECRET = config.get("api_secret")
OHLCV_INTERVAL = config.get("ohlcv_interval", "3m")
OHLCV_LIMIT = config.get("ohlcv_limit", 150)
SCAN_INTERVAL = config.get("scan_interval_minutes", 10)
RETRY_INTERVAL = config.get("retry_interval_seconds", 60)

# Inicialización de componentes
client = Client(API_KEY, API_SECRET)
scanner = MarketScanner()
data_loader = DataLoader(client)
order_manager = OrderManager()
buffer = SignalBuffer(config.get("signals_validated_path", "signals_validated.json"))

# ===============================
# 🔍 ESCANEO DE MERCADO
# ===============================
def ejecutar_escaneo():
    """
    Escanea el mercado, evalúa volatilidad y señales iniciales,
    y guarda las válidas en el buffer temporal.
    """
    print(f"\n🔁 Escaneando mercado completo — {datetime.now().strftime('%H:%M:%S')}")

    detectadas = 0
    total = 0

    try:
        sesgo = scanner.detectar_sesgo_mercado()
        intensidad = scanner.calcular_intensidad_mercado()
        print(f"🧭 Sesgo global: {sesgo.upper()} | 🌡️ Intensidad: {intensidad:.2f}")

        top_symbols = scanner.obtener_mejores_symbols()
        if not top_symbols:
            print("⚠️ No se encontraron símbolos destacados.")
            return 0

        for symbol in top_symbols:
            print(f"\n🔍 Analizando {symbol}...")
            df = data_loader.obtener_ohlcv(symbol, interval=OHLCV_INTERVAL, limit=OHLCV_LIMIT)
            total += 1

            if df is None or df.empty:
                print(f"⚠️ Sin datos recientes para {symbol}.")
                continue

            # Filtrar por volatilidad
            if not analizar_volatilidad(df):
                print(f"🌫️ {symbol}: Volatilidad insuficiente, omitiendo.")
                continue

            # Detección técnica
            resultado = detectar_senal(df)
            if resultado.get("entrada"):
                print(f"✅ Señal detectada en {symbol} ({resultado['estrategia']})")
                buffer.add(symbol, resultado["tipo"], resultado["estrategia"])
                detectadas += 1
            else:
                print(f"❌ Sin señal válida en {symbol}")

            print("-" * 45)
            time.sleep(1.2)

    except Exception as e:
        print(f"❌ Error en escaneo: {e}")
        send_critical("Sistema", f"Error durante escaneo: {e}")

    print(f"\n📊 Total analizadas: {total} | Señales detectadas: {detectadas}")
    return detectadas

# ===============================
# 🔄 CICLO OPERATIVO PRINCIPAL
# ===============================
def ciclo_principal():
    print(f"\n🕒 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔄 Iniciando ciclo operativo...")

    # 1️⃣ Escaneo técnico general
    total_detectadas = ejecutar_escaneo()

    # 2️⃣ Reevaluación + Validación Avanzada
    print("\n🤖 Reevaluando y validando confluencias...")
    reevaluar_senales()

    # 3️⃣ Monitoreo de operaciones activas
    print("\n📈 Monitoreando operaciones...")
    monitorear_operaciones()

    # 4️⃣ Reporte diario
    print("\n🧾 Generando reporte diario...")
    generar_reporte_diario()

    # 5️⃣ Notificación del ciclo
    try:
        resumen = (
            f"📊 *Resumen del Ciclo*\n"
            f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"🔍 Señales detectadas: {total_detectadas}\n"
        )
        send_info("Sistema", resumen)
    except Exception as e:
        print(f"⚠️ Error enviando resumen del ciclo: {e}")

    print("\n✅ Ciclo operativo finalizado.\n")

# ===============================
# 🔁 LOOP CONTINUO
# ===============================
if __name__ == "__main__":
    print("🚀 Iniciando bot de trading avanzado...\n")

    while True:
        try:
            ciclo_principal()
            print(f"⏳ Esperando {SCAN_INTERVAL} minutos antes del próximo ciclo...\n")
            time.sleep(SCAN_INTERVAL * 60)

        except KeyboardInterrupt:
            print("\n🛑 Ejecución detenida manualmente.")
            break

        except Exception as e:
            print(f"⚠️ Error crítico en el ciclo principal: {e}")
            send_critical("Sistema", f"Error crítico en el ciclo principal: {e}")
            print(f"⏳ Reintentando en {RETRY_INTERVAL} segundos...\n")
            time.sleep(RETRY_INTERVAL)
