# ==========================================================
# 📊 backtest/backtest_engine.py (versión final optimizada)
# ==========================================================
# Simulador de backtesting completo para estrategias del bot Binance.
# Evalúa señales, ejecuta operaciones simuladas con TP/SL dinámicos,
# calcula métricas de rentabilidad y genera reportes JSON y gráficos.
# ==========================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from binance.client import Client

# --- Configurar PATH del proyecto ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
print(f"📁 Proyecto cargado desde: {PROJECT_ROOT}")

# --- Imports internos ---
from data.data_loader import DataLoader
from indicators.signal_analyzer import detectar_senal
from core.signal_reevaluator import _compute_tp_sl

# === Carga de configuración ===
with open("config/settings.json") as f:
    config = json.load(f)

API_KEY = config["api_key"]
API_SECRET = config["api_secret"]
INTERVAL = config.get("ohlcv_interval", "1h")
CAPITAL_INICIAL = config.get("initial_capital", 1000.0)
RISK_PER_TRADE_USDT = config.get("risk_per_trade_usdt", 5.0)
PLOT_RESULTS = True

# === Cliente de Binance ===
client = Client(API_KEY, API_SECRET)
data_loader = DataLoader(client)


# ==========================================================
# 🎯 Clase principal del Backtester
# ==========================================================
class Backtester:
    def __init__(self, symbol="BTCUSDT", interval="1h", capital_inicial=CAPITAL_INICIAL, risk_per_trade_usdt=RISK_PER_TRADE_USDT):
        self.symbol = symbol
        self.interval = interval
        self.capital_inicial = capital_inicial
        self.capital = capital_inicial
        self.risk_per_trade_usdt = risk_per_trade_usdt
        self.trades = []
        self.results = {}

    # ==========================================================
    # 📦 Cargar datos históricos
    # ==========================================================
    def cargar_datos(self, limite=1000):
        """Carga velas desde Binance mediante DataLoader"""
        try:
            print(f"📦 Cargando datos históricos de {self.symbol} ({self.interval}) ...")
            df = data_loader.obtener_ohlcv(self.symbol, interval=self.interval, limit=limite)

            if df is None or df.empty:
                raise ValueError("No se obtuvieron datos del símbolo solicitado")

            print(f"✅ Datos cargados correctamente: {len(df)} velas recibidas.")
            return df
        except Exception as e:
            print(f"❌ Error al cargar datos: {e}")
            return pd.DataFrame()

    # ==========================================================
    # ⚙️ Simulación de operaciones
    # ==========================================================
    def ejecutar(self, limite=500, plot=True):
        df = self.cargar_datos(limite)
        if df.empty:
            print("⚠️ No hay datos para backtest.")
            return

        operaciones = []
        posicion = None

        for i in range(50, len(df)):  # saltamos las primeras velas por cálculo de indicadores
            sub_df = df.iloc[:i].copy()
            senal = detectar_senal(sub_df)
            precio = df.iloc[i]["close"]

            if not posicion and senal["entrada"]:
                tipo = senal["tipo"]

                # === Calcular ATR local ===
                high = sub_df["high"]
                low = sub_df["low"]
                close = sub_df["close"]
                prev_close = close.shift(1)
                tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean().iloc[-1]
                atr = float(atr) if not np.isnan(atr) else precio * 0.005  # fallback

                # === Calcular TP/SL ===
                tp, sl = _compute_tp_sl(precio, tipo, atr)

                # === Calcular tamaño de posición (por riesgo fijo en USDT) ===
                riesgo = self.risk_per_trade_usdt
                cantidad = riesgo / abs(precio - sl)

                posicion = {
                    "tipo": tipo,
                    "entrada": precio,
                    "tp": tp,
                    "sl": sl,
                    "cantidad": cantidad,
                    "indice": i
                }

                print(f"🎯 Entrada {tipo.upper()} en {precio:.2f} | TP={tp:.2f} | SL={sl:.2f}")
                continue

            if posicion:
                high = df.iloc[i]["high"]
                low = df.iloc[i]["low"]

                if posicion["tipo"] == "long":
                    if high >= posicion["tp"]:
                        ganancia = (posicion["tp"] - posicion["entrada"]) * posicion["cantidad"]
                        self.capital += ganancia
                        operaciones.append({"resultado": "win", "ganancia": ganancia})
                        print(f"✅ TP alcanzado: +{ganancia:.2f}")
                        posicion = None
                    elif low <= posicion["sl"]:
                        perdida = (posicion["entrada"] - posicion["sl"]) * posicion["cantidad"]
                        self.capital -= perdida
                        operaciones.append({"resultado": "loss", "ganancia": -perdida})
                        print(f"❌ SL alcanzado: -{perdida:.2f}")
                        posicion = None

                elif posicion["tipo"] == "short":
                    if low <= posicion["tp"]:
                        ganancia = (posicion["entrada"] - posicion["tp"]) * posicion["cantidad"]
                        self.capital += ganancia
                        operaciones.append({"resultado": "win", "ganancia": ganancia})
                        print(f"✅ TP alcanzado: +{ganancia:.2f}")
                        posicion = None
                    elif high >= posicion["sl"]:
                        perdida = (posicion["sl"] - posicion["entrada"]) * posicion["cantidad"]
                        self.capital -= perdida
                        operaciones.append({"resultado": "loss", "ganancia": -perdida})
                        print(f"❌ SL alcanzado: -{perdida:.2f}")
                        posicion = None

        self.trades = operaciones
        self._calcular_resultados()
        if plot:
            self._graficar_resultados(df)

    # ==========================================================
    # 📈 Cálculo de métricas finales
    # ==========================================================
    def _calcular_resultados(self):
        if not self.trades:
            print("⚠️ No se registraron operaciones.")
            return

        ganancias = [op["ganancia"] for op in self.trades]
        wins = sum(1 for g in ganancias if g > 0)
        losses = sum(1 for g in ganancias if g <= 0)
        total = len(ganancias)

        profit = sum(ganancias)
        avg_gain = np.mean([g for g in ganancias if g > 0]) if wins else 0
        avg_loss = np.mean([g for g in ganancias if g <= 0]) if losses else 0
        profit_factor = abs(avg_gain / avg_loss) if avg_loss != 0 else float("inf")
        win_rate = (wins / total) * 100 if total > 0 else 0
        drawdown = self._calcular_drawdown(ganancias)

        self.results = {
            "symbol": self.symbol,
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_profit": round(profit, 2),
            "max_drawdown": round(drawdown, 2),
            "capital_final": round(self.capital, 2)
        }

        print("\n📊 RESULTADOS DEL BACKTEST")
        print("========================================")
        for k, v in self.results.items():
            print(f"{k.replace('_', ' ').title()}: {v}")

        self._guardar_resultados()

    # ==========================================================
    # 📉 Cálculo de drawdown
    # ==========================================================
    def _calcular_drawdown(self, ganancias):
        balance = [self.capital_inicial]
        for g in ganancias:
            balance.append(balance[-1] + g)
        max_balance = np.maximum.accumulate(balance)
        dd = (max_balance - balance) / max_balance
        return np.max(dd) * 100 if len(dd) > 0 else 0

    # ==========================================================
    # 💾 Guardar resultados
    # ==========================================================
    def _guardar_resultados(self):
        folder = os.path.join(PROJECT_ROOT, "logs", "backtest")
        os.makedirs(folder, exist_ok=True)
        filename = f"backtest_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(folder, filename)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"💾 Resultados guardados en {path}")

    # ==========================================================
    # 📊 Gráfica de evolución del capital
    # ==========================================================
    def _graficar_resultados(self, df):
        plt.figure(figsize=(12, 6))
        closes = df["close"].values
        plt.plot(closes, label="Precio", alpha=0.4, color="gray")

        equity = [self.capital_inicial]
        capital = self.capital_inicial
        for op in self.trades:
            capital += op["ganancia"]
            equity.append(capital)

        plt.plot(np.linspace(0, len(closes), len(equity)), equity, color="green", linewidth=2, label="Evolución del Capital")
        plt.title(f"📊 Backtest {self.symbol} ({self.interval})")
        plt.xlabel("Velas")
        plt.ylabel("USDT")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ==========================================================
# 🧪 Ejecución directa
# ==========================================================
if __name__ == "__main__":
    bt = Backtester(symbol="BTCUSDT", interval="1h", capital_inicial=1000, risk_per_trade_usdt=5.0)
    bt.ejecutar(limite=500, plot=True)
