# utils/symbol_validator.py

import json
import os
from binance.exceptions import BinanceAPIException

CACHE_PATH = 'config/symbols_operables.json'

def validar_simbolo_operable(client, symbol, quote='USDT'):
    try:
        client.order_test(
            symbol=symbol,
            side='BUY',
            type='MARKET',
            quantity=10  # valor arbitrario, no se ejecuta
        )
        return True
    except BinanceAPIException as e:
        return False
    except Exception:
        return False

def obtener_symbols_operables(client):
    print("🔍 Obteniendo y validando símbolos operables...")

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
        if validar_simbolo_operable(client, symbol):
            print(f"✅ {symbol} operable")
            operables.append(symbol)
        else:
            print(f"❌ {symbol} no permitido por tu API")

    with open(CACHE_PATH, 'w') as f:
        json.dump(operables, f, indent=2)
    return operables

def cargar_symbols_operables():
    if not os.path.exists(CACHE_PATH):
        return []
    with open(CACHE_PATH, 'r') as f:
        return json.load(f)
