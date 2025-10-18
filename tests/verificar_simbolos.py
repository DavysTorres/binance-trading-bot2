from binance.client import Client

API_KEY = "JEUTikp38i3JR23VBIdFAyBu8Ag91GZzI0ShHon1dytV5Fn8pVvAwJ5Sfu1IaGG8"
API_SECRET = "O6oCE0yLgEsFhuhQPVwAyRwGL6TS6XG8grxQobSklNX1Esjf5dtfS8pwHVBh3v5E"

client = Client(API_KEY, API_SECRET)

def listar_simbolos_operables():
    try:
        info = client.get_exchange_info()
        posibles = [
            s for s in info['symbols']
            if s['status'] == 'TRADING'
            and s['isSpotTradingAllowed']
            and s['quoteAsset'] == 'USDT'
        ]

        print(f"🔍 Se encontraron {len(posibles)} símbolos operables (con USDT y en SPOT)")
        for s in posibles[:20]:  # muestra solo los primeros 20
            print(f"✅ {s['symbol']}")

    except Exception as e:
        print("❌ Error al obtener símbolos:", e)

if __name__ == "__main__":
    listar_simbolos_operables()
