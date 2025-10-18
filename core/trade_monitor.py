# core/trade_monitor.py
# ===============================================================
# 🧠 Monitor Avanzado de Operaciones
# ---------------------------------------------------------------
# Evalúa posiciones abiertas en base a precio, volatilidad,
# sesgo global y score técnico. Gestiona trailing y cierra
# dinámicamente según confluencia inversa o SL/TP.
# ===============================================================

import os, sys, json, time
import pandas as pd
from datetime import datetime
from binance.client import Client

# Acceso a los módulos locales
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_loader import DataLoader
from indicators.signal_analyzer import detectar_senal
from ai.advanced_signal_validator import validar_confluencia
from indicators.ema import calcular_ema
from indicators.rsi import calcular_rsi
from indicators.volumen import calcular_volumen_promedio
from trading.order_trailing_manager import ajustar_trailing_stop
from core.report_manager import registrar_evento
from core.notification_manager import send_info, send_warning, send_critical
from modules.market_scanner_module import MarketScanner

# ===============================================================
# 🔧 Configuración inicial
# ===============================================================
with open("config/settings.json") as f:
    config = json.load(f)

API_KEY = config["api_key"]
API_SECRET = config["api_secret"]
OPEN_ORDERS_FILE = config.get("open_orders_file", "logs/open_orders.json")
OHLCV_INTERVAL = config.get("ohlcv_interval", "3m")
OHLCV_LIMIT = config.get("ohlcv_limit", 150)
LOOP_INTERVAL = config.get("trade_monitor_loop_seconds", 120)

client = Client(API_KEY, API_SECRET)
data_loader = DataLoader(client)

# ===============================================================
# 🧩 Utilidades
# ===============================================================
def cargar_ordenes_abiertas():
    if not os.path.exists(OPEN_ORDERS_FILE):
        return []
    with open(OPEN_ORDERS_FILE, "r") as f:
        return json.load(f)

def guardar_ordenes_abiertas(ordenes):
    with open(OPEN_ORDERS_FILE, "w") as f:
        json.dump(ordenes, f, indent=4)

def obtener_precio_actual(symbol):
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])
    except Exception:
        return None

def calcular_duracion(timestamp_in):
    if not timestamp_in:
        return "N/A"
    s = str(timestamp_in).strip()
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S"):
        try:
            ts = datetime.strptime(s, fmt)
            diff = datetime.now() - ts
            d, r = divmod(diff.seconds, 3600)
            m, _ = divmod(r, 60)
            return f"{diff.days}d {d}h {m}m"
        except Exception:
            continue
    return "N/A"

# ===============================================================
# ⚙️ Monitoreo Avanzado
# ===============================================================
def monitorear_operaciones():
    ordenes = cargar_ordenes_abiertas()
    if not ordenes:
        print("📭 No hay operaciones abiertas para monitorear.")
        return

    try:
        scanner = MarketScanner()
        sesgo_global = scanner.detectar_sesgo_mercado()
        intensidad = scanner.calcular_intensidad_mercado()
    except Exception as e:
        print(f"⚠️ No se pudo obtener sesgo global: {e}")
        sesgo_global = "neutral"
        intensidad = 0.5

    print(f"\n🧭 Sesgo Global: {sesgo_global.upper()} | Intensidad: {intensidad:.2f}")

    nuevas_ordenes = []

    for orden in ordenes:
        symbol = orden.get("symbol")
        side = orden.get("side", "").lower()
        precio_entrada = float(orden.get("precio_entrada", 0))
        sl = float(orden.get("stop_loss", 0))
        tp = float(orden.get("take_profit", 0))
        cantidad = float(orden.get("cantidad", 0))
        score = float(orden.get("score", 0.5))
        timestamp_str = orden.get("timestamp", "")
        precio_actual = obtener_precio_actual(symbol)

        if not precio_actual:
            print(f"⚠️ No se pudo obtener precio de {symbol}")
            nuevas_ordenes.append(orden)
            continue

        variacion_pct = ((precio_actual - precio_entrada) / precio_entrada) * 100
        pnl = cantidad * (precio_actual - precio_entrada)
        duracion = calcular_duracion(timestamp_str)

        print(f"\n📊 {symbol} | {side.upper()} | Δ%={variacion_pct:.2f} | Δ${pnl:.2f} | Score={score:.2f}")

        # ======================
        # 🧠 Reevaluación técnica
        # ======================
        df = data_loader.obtener_ohlcv(symbol, interval=OHLCV_INTERVAL, limit=OHLCV_LIMIT)
        if df is None or df.empty:
            print(f"⚠️ Datos insuficientes para {symbol}")
            nuevas_ordenes.append(orden)
            continue

        df = calcular_ema(calcular_rsi(df))
        df["volumen_movil_20"] = calcular_volumen_promedio(df, 20)
        resultado = detectar_senal(df)
        tipo_actual = resultado.get("tipo", "neutro")
        estrategia = resultado.get("estrategia", "indefinida")

        # Validar con confluencia avanzada
        val = validar_confluencia(df.copy(), tipo_actual)
        confluencia_score = float(val.get("score", 0.0))
        ok_confluencia = bool(val.get("ok", False))

        # ===================================
        # 🎯 Trailing adaptativo
        # ===================================
        try:
            ajustar_trailing_stop(symbol, precio_actual, orden)
        except Exception as e:
            print(f"⚠️ Error ajustando trailing: {e}")

        # ===================================
        # 📉 Reglas de cierre inteligente
        # ===================================
        cierre = None

        # 1️⃣ Stop Loss o Take Profit
        if (side == "buy" and precio_actual <= sl) or (side == "sell" and precio_actual >= sl):
            cierre = ("SL", "Stop Loss alcanzado")
        elif (side == "buy" and precio_actual >= tp) or (side == "sell" and precio_actual <= tp):
            cierre = ("TP", "Take Profit alcanzado")

        # 2️⃣ Señal contraria confirmada
        elif not ok_confluencia and confluencia_score < 0.45:
            cierre = ("REV", "Confluencia contraria confirmada")

        # 3️⃣ Pérdida prolongada
        elif variacion_pct < -2 and duracion.startswith("0d"):
            cierre = ("DRAWDOWN", "Pérdida temprana detectada")

        # ===================================
        # 📈 Evaluación y registro
        # ===================================
        if cierre:
            tipo_cierre, motivo = cierre
            color = "🔴" if tipo_cierre in ["SL", "DRAWDOWN"] else "🟢" if tipo_cierre == "TP" else "🟠"
            msg = f"{color} {symbol} cerrado — {motivo}. Δ%={variacion_pct:.2f} | Score={score:.2f} | Confluencia={confluencia_score:.2f}"

            # Notificaciones
            if tipo_cierre == "TP":
                send_info(symbol, msg)
            elif tipo_cierre == "SL":
                send_critical(symbol, msg)
            else:
                send_warning(symbol, msg)

            registrar_evento(symbol, "monitoreo", "cerrado", tipo_cierre == "TP", msg)
            cerrar_operacion(symbol, motivo)
            continue

        # ===================================
        # 🟢 Mantener operación activa
        # ===================================
        orden["ultimo_score"] = score
        orden["confluencia_score"] = confluencia_score
        orden["variacion_pct"] = round(variacion_pct, 2)
        orden["pnl"] = round(pnl, 3)
        orden["duracion"] = duracion
        nuevas_ordenes.append(orden)

        print(f"✅ {symbol} activo. Δ%={variacion_pct:.2f} | Conf={confluencia_score:.2f} | Estrategia={estrategia}")

    guardar_ordenes_abiertas(nuevas_ordenes)
    print("\n📈 Monitoreo avanzado completado.")

# ===============================================================
# 🔒 Cierre de operación
# ===============================================================
def cerrar_operacion(symbol, motivo="manual"):
    if not os.path.exists(OPEN_ORDERS_FILE):
        return
    with open(OPEN_ORDERS_FILE, "r") as f:
        ordenes = json.load(f)
    ordenes = [o for o in ordenes if o.get("symbol") != symbol]
    with open(OPEN_ORDERS_FILE, "w") as f:
        json.dump(ordenes, f, indent=4)
    print(f"🔒 Operación en {symbol} cerrada ({motivo}).")

# ===============================================================
# ⏱️ Loop principal
# ===============================================================
def loop_monitoreo():
    while True:
        print(f"\n⏱️ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Ciclo de monitoreo iniciado...")
        monitorear_operaciones()
        print(f"🕒 Esperando {LOOP_INTERVAL} segundos antes del siguiente ciclo...")
        time.sleep(LOOP_INTERVAL)

# ===============================================================
# 🚀 Ejecución directa
# ===============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monitor técnico avanzado de operaciones abiertas.")
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--loop', action='store_true')
    args = parser.parse_args()
    if args.once:
        monitorear_operaciones()
    elif args.loop:
        loop_monitoreo()
    else:
        print("❗ Usa --once o --loop para ejecutar correctamente.")
