# trading/order_tracker.py

import os
import json
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trading.binance_connector import BinanceConnector

class OrderTracker:
    def __init__(self, config_path='config/settings.json'):
        self.api = BinanceConnector(config_path)
        self._load_config(config_path)

    def _load_config(self, path):
        with open(path, 'r') as f:
            config = json.load(f)
            self.trailing_enabled = config.get("trailing_stop_enabled", True)
            self.trailing_pct = config.get("trailing_stop_pct", 0.01)
            self.market_type = config.get("market_type", "SPOT")

    def revisar_ordenes(self, logs_folder="logs"):
        print("🔍 Revisando órdenes abiertas...")

        for filename in os.listdir(logs_folder):
            if not filename.endswith(".json"):
                continue

            ruta_log = os.path.join(logs_folder, filename)
            with open(ruta_log, 'r') as f:
                data = json.load(f)

            if data.get("cerrada", False):
                continue  # ya fue cerrada

            symbol = data['symbol']
            cantidad = data['cantidad']
            entrada = data['precio_entrada']
            sl = data['stop_loss']
            tp = data['take_profit']

            precio_actual = self.api.get_current_price(symbol)
            if not precio_actual:
                continue

            print(f"🔄 {symbol} | Entrada: {entrada} | Actual: {precio_actual} | TP: {tp} | SL: {sl}")

            # 🧠 Trailing Stop
            if self.trailing_enabled and precio_actual > entrada:
                nuevo_sl = round(precio_actual * (1 - self.trailing_pct), 4)
                if nuevo_sl > sl:
                    print(f"📈 Ajustando SL dinámicamente: {sl} → {nuevo_sl}")
                    data['stop_loss'] = nuevo_sl
                    self._guardar_log(data, ruta_log)
                    sl = nuevo_sl

            # 🟥 Verificar Stop Loss
            if precio_actual <= sl:
                print(f"🛑 Precio alcanzó SL. Cerrando operación {symbol}")
                self._cerrar_operacion(symbol, cantidad, "SL", ruta_log)
                continue

            # 🟩 Verificar Take Profit
            if precio_actual >= tp:
                print(f"🎯 Precio alcanzó TP. Cerrando operación {symbol}")
                self._cerrar_operacion(symbol, cantidad, "TP", ruta_log)
                continue

    def _cerrar_operacion(self, symbol, cantidad, motivo, ruta_log):
        print(f"📤 Ejecutando orden de cierre ({motivo}) para {symbol}...")

        try:
            cierre = self.api.place_market_order("sell", quantity=cantidad, symbol=symbol)
            if cierre:
                print(f"✅ Orden cerrada correctamente.")
        except Exception as e:
            print(f"❌ Error al cerrar operación: {e}")

        # Marcar log como cerrado
        with open(ruta_log, 'r') as f:
            data = json.load(f)

        data['cerrada'] = True
        data['motivo_cierre'] = motivo
        data['precio_cierre'] = self.api.get_current_price(symbol)
        data['timestamp_cierre'] = int(time.time())

        self._guardar_log(data, ruta_log)

    def _guardar_log(self, data, ruta):
        with open(ruta, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"📝 Log actualizado: {ruta}")


if __name__ == "__main__":
    tracker = OrderTracker()
    while True:
        tracker.revisar_ordenes()
        time.sleep(30)  # cada 30 segundos
