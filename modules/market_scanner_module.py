# modules/market_scanner_module.py

from trading.binance_connector import BinanceConnector
from strategies.multi_timeframe import analizar_curva
from indicators.volatility_analyzer import analizar_volatilidad
from utils.lote_index_manager import cargar_lote_index, guardar_lote_index
import time
import json
import pandas as pd


class MarketScanner:
    def __init__(self, config_path='config/settings.json'):
        self.api = BinanceConnector(config_path)
        self._load_config(config_path)
        self.lote_index = 0
        self.simbolos_operables = self.get_symbols_operables()
        self.market_bias = "neutral"  # 🧭 Sesgo global del mercado

    def _load_config(self, path):
        with open(path, 'r') as f:
            config = json.load(f)
            self.max_symbols = config.get("max_symbols", 10)
            self.min_volume = config.get("min_volume_usdt", 1000000)
            self.analysis_interval = config.get("scan_interval_minutes", 5)
            self.interval = config.get("analysis_interval", "3m")
            self.limit = config.get("klines_limit", 100)
            self.lote_size = config.get("rotation_lote_size", 20)

    # -----------------------------------------
    # 🔄 Carga y filtrado de símbolos operables
    # ----------------------------------------
    def get_symbols_operables(self):
        print("🔄 Cargando lista actualizada desde Binance directamente...")
        exchange_info = self.api.client.get_exchange_info()
        symbols = [
            s['symbol'] for s in exchange_info['symbols']
            if s['status'] == 'TRADING'
            and s['isSpotTradingAllowed']
            and s['quoteAsset'] == 'USDT'
            and 'MARKET' in s['orderTypes']
        ]
        return sorted(symbols)

    # -----------------------------------------
    # 🧭 Detección automática del sesgo global
    # -----------------------------------------
    def detectar_sesgo_mercado(self):
        tickers = self.api.client.get_ticker()
        positivos, negativos = 0, 0
        total = 0

        for t in tickers:
            if not t["symbol"].endswith("USDT"):
                continue
            cambio = float(t.get("priceChangePercent", 0))
            if cambio > 0:
                positivos += 1
            elif cambio < 0:
                negativos += 1
            total += 1

        if total == 0:
            self.market_bias = "neutral"
            return "neutral"

        porc_positivos = positivos / total * 100
        porc_negativos = negativos / total * 100

        if porc_positivos >= 60:
            self.market_bias = "bullish"
        elif porc_negativos >= 60:
            self.market_bias = "bearish"
        else:
            self.market_bias = "neutral"

        icon = "🟢" if self.market_bias == "bullish" else "🔴" if self.market_bias == "bearish" else "⚪"
        print(f"{icon} Sesgo global detectado: {self.market_bias.upper()} ({porc_positivos:.1f}% alcistas / {porc_negativos:.1f}% bajistas)")
        return self.market_bias

    # -----------------------------------------
    # 📈 Selección dinámica de símbolos
    # -----------------------------------------
    def obtener_mejores_symbols(self):
        # Evaluar el sesgo antes de buscar oportunidades
        self.detectar_sesgo_mercado()
        self.market_intensity = self.calcular_intensidad_mercado()


        total_lotes = len(self.simbolos_operables) // self.lote_size + 1
        self.lote_index = cargar_lote_index()

        inicio = self.lote_index * self.lote_size
        fin = inicio + self.lote_size
        lote = self.simbolos_operables[inicio:fin]

        self.lote_index = (self.lote_index + 1) % total_lotes
        guardar_lote_index(self.lote_index)

        tickers = self.api.client.get_ticker()
        symbols_set = set([s.upper() for s in lote])
        candidatos = []

        for t in tickers:
            sym = t["symbol"].upper()
            if sym not in symbols_set or not sym.endswith("USDT"):
                continue

            vol = float(t.get("quoteVolume", 0))
            cambio = float(t.get("priceChangePercent", 0))
            if vol < self.min_volume:
                continue

            tendencia = "LONG" if cambio > 0 else "SHORT"
            # Filtrar según el sesgo del mercado
            if self.market_bias == "bullish" and cambio < 0:
                continue
            if self.market_bias == "bearish" and cambio > 0:
                continue

            candidatos.append({
                "symbol": sym,
                "priceChangePercent": cambio,
                "quoteVolume": vol,
                "tendencia": tendencia
            })

        # Ordenar por magnitud del movimiento
        ordenados = sorted(candidatos, key=lambda x: abs(x["priceChangePercent"]), reverse=True)
        top = ordenados[:self.max_symbols]

        top_symbols = [s["symbol"] for s in top]
        print(f"🔍 Lote {self.lote_index}/{total_lotes} - Top {self.max_symbols} símbolos ({self.market_bias.upper()}): {top_symbols}")
        for s in top:
            dir_icon = "🚀" if s["tendencia"] == "LONG" else "📉"
            print(f"  {dir_icon} {s['symbol']} ({s['tendencia']}) cambio: {s['priceChangePercent']}% volumen: {int(s['quoteVolume'])}")

        return [s["symbol"] for s in top]

            

    def escanear_mercado(self):
        print("\n🔁 Escaneando mercado para oportunidades...")
        symbols = self.obtener_mejores_symbols()
        oportunidades = []

        for symbol in symbols:
            try:
                print(f"🔍 Analizando {symbol}...")
                datos_raw = self.api.get_klines(symbol=symbol, interval=self.interval, limit=self.limit)

                # Validación inicial
                if not datos_raw or len(datos_raw) < self.limit:
                    print(f"⚠️ No se pudo obtener suficientes datos de {symbol}, saltando...")
                    continue

                # Convertir a DataFrame con las columnas adecuadas
                datos = pd.DataFrame(datos_raw, columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                ])

                # Conversión de columnas numéricas necesarias
                for col in ["open", "high", "low", "close", "volume"]:
                    datos[col] = pd.to_numeric(datos[col], errors='coerce')

                # Seleccionar columnas útiles
                datos = datos[["timestamp", "open", "high", "low", "close", "volume"]]
                print(f"📊 Columnas de {symbol}: {list(datos.columns)}")

                print(f"📈 Evaluando volatilidad de {symbol}...")
                if not analizar_volatilidad(datos):
                    print(f"🌫️ Volatilidad insuficiente en {symbol}, omitiendo...")
                    continue

                # 🔍 Verificación de señal técnica
                if analizar_curva(datos):
                    print(f"✅ Señal confirmada en {symbol}")
                    oportunidades.append(symbol)
                else:
                     print(f"⏭️ Sin señal técnica válida en {symbol}")

            except Exception as e:
                print(f"⚠️ Error al analizar {symbol}: {e}")

            print("-" * 40)

        return oportunidades
    
    def calcular_intensidad_mercado(self):
        """Evalúa qué tan fuerte es la tendencia global (de 0 a 1)."""
        tickers = self.api.client.get_ticker()
        variaciones = [
            abs(float(t.get("priceChangePercent", 0)))
            for t in tickers if t["symbol"].endswith("USDT")
        ]
        if not variaciones:
            return 0.0

        promedio = sum(variaciones) / len(variaciones)
        intensidad = min(promedio / 5, 1.0)  # Normalizamos con un límite superior
        print(f"🌡️ Intensidad de mercado: {intensidad:.2f}")
        return intensidad



if __name__ == "__main__":
    scanner = MarketScanner()
    while True:
        oportunidades = scanner.escanear_mercado()
        if oportunidades:
            print(f"📈 Oportunidades encontradas: {oportunidades}")
            # Aquí se podría invocar order_manager.ejecutar_orden_con_sl_tp(symbol)
        else:
            print("😴 No se encontraron oportunidades.")

        time.sleep(60 * scanner.analysis_interval)
