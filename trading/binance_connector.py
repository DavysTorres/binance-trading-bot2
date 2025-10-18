# trading/binance_connector.py

from binance.client import Client
from binance.enums import *
import json
import os

class BinanceConnector:
    def __init__(self, config_path='config/settings.json'):
        self._load_config(config_path)
        self.client = Client(self.api_key, self.api_secret)

    def _load_config(self, path):
        with open(path, 'r') as f:
            config = json.load(f)
            self.api_key = config["api_key"]
            self.api_secret = config["api_secret"]
            self.default_symbol = config.get("default_symbol", "BTCUSDT")

    # Obtener datos OHLCV
    def get_klines(self, symbol=None, interval='1m', limit=100):
        symbol = symbol or self.default_symbol
        try:
            return self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        except Exception as e:
            print(f"❌ Error al obtener velas: {e}")
            return []

    # Obtener precio actual
    def get_current_price(self, symbol=None):
        symbol = symbol or self.default_symbol
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            print(f"❌ Error al obtener el precio: {e}")
            return None

    # Obtener balance disponible de un activo (por defecto USDT)
    def get_balance(self, asset="USDT"):
        try:
            balances = self.client.get_asset_balance(asset=asset)
            return float(balances['free'])
        except Exception as e:
            print(f"❌ Error al obtener balance de {asset}: {e}")
            return None

    # Crear orden de mercado
    def place_market_order(self, side, quantity, symbol=None):
        symbol = symbol or self.default_symbol
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=SIDE_BUY if side.lower() == "buy" else SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            print(f"✅ Orden de mercado {side.upper()} ejecutada: {order}")
            return order
        except Exception as e:
            print(f"❌ Error al ejecutar orden de mercado: {e}")
            return None

    # Crear orden limitada con TP/SL
    def place_limit_order(self, side, quantity, price, symbol=None):
        symbol = symbol or self.default_symbol
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=SIDE_BUY if side.lower() == "buy" else SIDE_SELL,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=quantity,
                price=str(price)
            )
            print(f"✅ Orden LIMIT {side.upper()} colocada: {order}")
            return order
        except Exception as e:
            print(f"❌ Error al colocar orden LIMIT: {e}")
            return None
        
    def is_symbol_executable(self, symbol):
        try:
            info = self.client.get_exchange_info()
            symbol_info = next((s for s in info['symbols'] if s['symbol'] == symbol), None)

            if not symbol_info:
                return False

            return (
                symbol_info.get('status') == 'TRADING'
                and symbol_info.get('isSpotTradingAllowed', False)
                and symbol_info.get('quoteAsset') == 'USDT'
                and 'MARKET' in symbol_info.get('orderTypes', [])
            )
        except Exception as e:
            print(f"⚠️ Error verificando si {symbol} es ejecutable: {e}")
            return False
        
    def get_lot_size_filter(self, symbol):
        info = self.client.get_symbol_info(symbol)
        if info:
            for f in info["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    return {
                        "minQty": float(f["minQty"]),
                        "maxQty": float(f["maxQty"]),
                        "stepSize": float(f["stepSize"])
                    }
        return None
    
    def get_notional_filter(self, symbol):
        info = self.client.get_symbol_info(symbol)
        if info:
            for f in info["filters"]:
                if f["filterType"] == "MIN_NOTIONAL":
                    return {
                        "minNotional": float(f["minNotional"])
                    }
        return None



