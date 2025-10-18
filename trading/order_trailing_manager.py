# trading/order_trailing_manager.py

import os
import json
import time
import glob
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.settings import load_settings
from utils.logger import log
from utils.price_fetcher import get_price

# -----------------------
# 🔧 CONFIGURACIÓN
# -----------------------
settings = load_settings()
TRAILING_STOP_PCT = settings.get("trailing_stop_pct", 0.01)
LOGS_DIR = "logs"

# -----------------------
# 📦 FUNCIONES EXISTENTES
# -----------------------
def cargar_ordenes_simuladas():
    archivos_ordenes = glob.glob(f"{LOGS_DIR}/orden_*.json")
    ordenes = []
    for archivo in archivos_ordenes:
        with open(archivo, "r") as f:
            try:
                data = json.load(f)
                if data.get("simulada", False):
                    ordenes.append(data)
            except Exception as e:
                log(f"❌ Error al cargar {archivo}: {str(e)}")
    return ordenes


def cargar_trailing(symbol, timestamp):
    ruta = f"{LOGS_DIR}/trailing_{symbol}_{timestamp}.json"
    if os.path.exists(ruta):
        with open(ruta, "r") as f:
            return json.load(f)
    else:
        return None


def guardar_trailing(symbol, timestamp, datos):
    ruta = f"{LOGS_DIR}/trailing_{symbol}_{timestamp}.json"
    with open(ruta, "w") as f:
        json.dump(datos, f, indent=4)


def revisar_trailing_stop():
    log("🔍 Revisando órdenes simuladas para trailing stop...\n")
    ordenes = cargar_ordenes_simuladas()
    for orden in ordenes:
        symbol = orden["symbol"]
        precio_entrada = orden["precio_entrada"]
        sl_actual = orden["stop_loss"]
        timestamp = orden["timestamp"]
        cantidad = orden["cantidad"]

        precio_actual = get_price(symbol)
        if not precio_actual:
            log(f"⚠️ No se pudo obtener precio para {symbol}")
            continue

        trailing_data = cargar_trailing(symbol, timestamp)

        # Inicializa el máximo si es la primera vez
        if not trailing_data:
            trailing_data = {
                "symbol": symbol,
                "timestamp": timestamp,
                "cantidad": cantidad,
                "precio_entrada": precio_entrada,
                "max_price": precio_actual,
                "stop_loss": sl_actual
            }

        max_price = trailing_data["max_price"]

        if precio_actual > max_price:
            max_price = precio_actual
            nuevo_sl = round(max_price * (1 - TRAILING_STOP_PCT), 6)

            # Solo actualizar si supera al SL anterior
            if nuevo_sl > trailing_data["stop_loss"]:
                trailing_data["max_price"] = max_price
                trailing_data["stop_loss"] = nuevo_sl
                guardar_trailing(symbol, timestamp, trailing_data)
                log(f"🔁 {symbol}: SL actualizado a {nuevo_sl} con precio {precio_actual}")
            else:
                log(f"➖ {symbol}: Nuevo SL ({nuevo_sl}) menor o igual al anterior ({trailing_data['stop_loss']})")
        else:
            log(f"📉 {symbol}: Precio actual {precio_actual} no supera máximo {max_price}")

    log("\n✅ Revisión de trailing finalizada.\n")

# -----------------------
# 🧩 NUEVA FUNCIÓN PÚBLICA
# -----------------------
def ajustar_trailing_stop(symbol, precio_actual, orden):
    """
    Ajusta el trailing stop dinámicamente en órdenes reales o en seguimiento.
    Compatible con core/trade_monitor.py.
    """
    try:
        sl_actual = float(orden.get("stop_loss", 0))
        trailing_pct = orden.get("trailing_stop_pct", TRAILING_STOP_PCT)
        side = orden.get("side", "").lower()

        if not sl_actual or not precio_actual:
            log(f"⚠️ Datos incompletos para {symbol}: SL o precio actual faltante.")
            return orden

        if side == "buy":
            nuevo_sl = max(sl_actual, round(precio_actual * (1 - trailing_pct), 6))
        else:
            nuevo_sl = min(sl_actual, round(precio_actual * (1 + trailing_pct), 6))

        if nuevo_sl != sl_actual:
            orden["stop_loss"] = nuevo_sl
            log(f"🔁 {symbol}: Stop Loss ajustado dinámicamente a {nuevo_sl}")
        else:
            log(f"⏸️ {symbol}: Sin cambio en Stop Loss ({sl_actual})")

        return orden

    except Exception as e:
        log(f"❌ Error ajustando trailing stop en {symbol}: {str(e)}")
        return orden


# -----------------------
# 🚀 EJECUCIÓN DIRECTA
# -----------------------
if __name__ == "__main__":
    revisar_trailing_stop()
