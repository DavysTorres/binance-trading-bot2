# data/data_loader.py

from binance.client import Client
import pandas as pd
import time

class DataLoader:
    def __init__(self, client):
        self.client = client

    def obtener_ohlcv(self, symbol, interval="1m", limit=100):
        try:
            raw = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)

            # 🔎 Validar respuesta antes de crear DataFrame
            if not raw or len(raw) == 0:
                print(f"⚠️ Binance devolvió una lista vacía para {symbol}")
                return pd.DataFrame()

            # 🔧 Crear DataFrame completo con todas las columnas
            df = pd.DataFrame(raw, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ])

            # 🧩 Validar que todas las columnas existan
            expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                print(f"⚠️ Columnas faltantes en datos de {symbol}: {missing}")
                return pd.DataFrame()

            # 🧠 Limpieza y conversión segura
            df = df[expected_cols].copy()
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # 🧩 Eliminar duplicados y NaN por seguridad
            df = df.dropna().drop_duplicates(subset=['timestamp'])
            if len(df) < 5:
                print(f"⚠️ Muy pocos datos OHLCV ({len(df)}) para {symbol}. Se omite.")
                return pd.DataFrame()

            return df

        except Exception as e:
            print(f"❌ Error al obtener OHLCV para {symbol}: {e}")
            return pd.DataFrame()
