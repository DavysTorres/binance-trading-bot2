# test_connection.py

from trading.binance_connector import BinanceConnector
import json

def test_binance_connection():
    print("🔄 Inicializando conexión con Binance...")

    connector = BinanceConnector()

    # 1. Obtener precio actual
    price = connector.get_current_price()
    print(f"🟡 Precio actual BTCUSDT: {price}")

    # 2. Obtener velas (OHLCV)
    candles = connector.get_klines(limit=5)
    print("🟣 Últimas 5 velas:")
    for c in candles:
        print(f"  ⏱️ {c[0]} | 🟩 Open: {c[1]} | 🟥 Close: {c[4]}")

    # 3. Balance disponible
    usdt_balance = connector.get_balance("USDT")
    print(f"💰 Balance USDT: {usdt_balance}")

    # 4. Simular orden de mercado (comentada por seguridad)
    print("⚠️ Simulación de orden de compra:")
    print("🔒 Por seguridad, esta orden está comentada. Descomenta para ejecutarla.")

    # connector.place_market_order("buy", 0.001)

if __name__ == "__main__":
    test_binance_connection()
