import os
import json
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Ajusta con tu ruta o credenciales desde settings.json
with open('config/settings.json', 'r') as f:
    settings = json.load(f)

API_KEY = settings["api_key"]
API_SECRET = settings["api_secret"]
CACHE_PATH = 'config/symbols_operables.json'

client = Client(API_KEY, API_SECRET)

def validar_simbolo_operable(symbol):
    try:
        # Usamos una cantidad arbitraria, no se ejecuta en real
        client.create_test_order(
            symbol=symbol,
            side='BUY',
            type='MARKET',
            quantity=10
        )
        return True
    except BinanceAPIException:
        return False
    except Exception as e:
        print(f"⚠️ Otro error validando {symbol}: {e}")
        return False

def generar_symbols_operables():
    print("🔍 Generando lista de símbolos operables...")

    info = client.get_exchange_info()
    posibles = [
        s for s in info['symbols']
        if s['status'] == 'TRADING'
        and s['isSpotTradingAllowed']
        and s['quoteAsset'] == 'USDT'
    ]

    operables = []
    for s in posibles:
        symbol = s['symbol']
        print(f"⏳ Probando {symbol}...", end="")
        if validar_simbolo_operable(symbol):
            print("✅ OK")
            operables.append(symbol)
        else:
            print("❌ NO")

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(operables, f, indent=2)

    print(f"\n✅ Símbolos operables guardados en {CACHE_PATH} ({len(operables)} símbolos).")

if __name__ == "__main__":
    generar_symbols_operables()
