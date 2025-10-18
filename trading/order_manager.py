# trading/order_manager.py
# ===============================================================
# 🧠 OrderManager Avanzado con gestión adaptativa de riesgo
# ---------------------------------------------------------------
# Ajusta el tamaño, SL/TP y trailing dinámico según el score técnico.
# ===============================================================

from datetime import datetime
from decimal import Decimal, ROUND_DOWN
import os, json, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading.binance_connector import BinanceConnector
from trading.utils_legacy import calcular_volatilidad
from rich.console import Console
from rich.table import Table

# ===============================================================
# ⚙️ Configuración inicial
# ===============================================================
try:
    with open("config/settings.json", "r") as f:
        CONFIG = json.load(f)
except Exception:
    CONFIG = {}

OPEN_ORDERS_FILE = CONFIG.get("open_orders_file", "logs/open_orders.json")

def _append_open_order(record: dict):
    os.makedirs(os.path.dirname(OPEN_ORDERS_FILE), exist_ok=True)
    data = []
    if os.path.exists(OPEN_ORDERS_FILE):
        try:
            with open(OPEN_ORDERS_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            data = []
    data = [o for o in data if o.get("symbol") != record.get("symbol")] + [record]
    with open(OPEN_ORDERS_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ===============================================================
# 🔧 OrderManager
# ===============================================================
class OrderManager:
    def __init__(self, config_path='config/settings.json'):
        self.api = BinanceConnector(config_path)
        self._load_config(config_path)

    # -------------------------------------------------------------
    # 📂 Carga de configuración
    # -------------------------------------------------------------
    def _load_config(self, path):
        with open(path, 'r') as f:
            config = json.load(f)
        self.symbol = config.get("default_symbol", "BTCUSDT")
        self.risk_pct = float(config.get("risk_percent", 1.0))
        self.sl_pct = float(config.get("stop_loss_pct", 0.02))
        self.tp_pct = float(config.get("take_profit_pct", 0.03))
        self.trailing_stop = config.get("trailing_stop_enabled", True)
        self.trailing_stop_pct = float(config.get("trailing_stop_pct", 0.01))
        self.use_volatility = config.get("use_volatility_sl_tp", True)
        self.sim_mode = bool(config.get("sim_mode", False))
        self.capital_per_trade = float(config.get("capital_per_trade", 15.0))
        self.market_type = config.get("market_type", "SPOT")
        self.logs_dir = config.get("logs_dir", "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 🧮 Utilidades
    # -------------------------------------------------------------
    def _align_qty(self, qty, step):
        if not step:
            return float(round(qty, 6))
        q = Decimal(str(qty))
        s = Decimal(str(step))
        aligned = (q // s) * s
        return float(aligned.quantize(Decimal("1e-10"), rounding=ROUND_DOWN))

    def _guardar_log(self, data, tipo="orden"):
        filename = os.path.join(self.logs_dir, f"{tipo}_{data['timestamp']}.json")
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"📝 Log guardado → {filename}")

    # ===============================================================
    # 🧪 SIMULADOR DINÁMICO (adaptativo según score)
    # ===============================================================
    def _simular_orden_avanzada(self, symbol, side, entry, score=0.5,
                                qty=None, sl=None, tp=None,
                                motivo="Simulación avanzada", confluencia=None):
        """
        Simula una orden adaptando TP/SL y trailing según score técnico.
        """
        score = max(0.1, min(score, 1.0))  # rango seguro [0.1–1.0]
        confianza = round(score * 100, 1)

        if qty is None:
            qty = round(self.capital_per_trade / entry, 6)

        # Si no se proporcionan SL/TP → calcular adaptativamente
        if sl is None or tp is None:
            base_tp = self.tp_pct * (1 + (score - 0.5))
            base_sl = self.sl_pct * (1 - (score - 0.5))
            if side.lower() == "buy":
                sl = round(entry * (1 - base_sl), 8)
                tp = round(entry * (1 + base_tp), 8)
            else:
                sl = round(entry * (1 + base_sl), 8)
                tp = round(entry * (1 - base_tp), 8)

        # Trailing adaptativo
        trailing_pct = self.trailing_stop_pct * (1 - (score - 0.5))
        trailing_pct = max(0.003, round(trailing_pct, 4))

        data = {
            "modo": "simulado",
            "symbol": symbol,
            "side": side,
            "precio_entrada": entry,
            "cantidad": qty,
            "stop_loss": sl,
            "take_profit": tp,
            "trailing_stop_pct": trailing_pct,
            "score": round(score, 3),
            "confianza_%": confianza,
            "confluencia_score": round(confluencia or 0.0, 3),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "motivo": motivo
        }

        print(f"🧠 Simulación avanzada {symbol} {side.upper()} | "
              f"Score={score:.2f} ({confianza:.1f}%) | TP={tp} | SL={sl}")

        self._guardar_log(data)
        if self.trailing_stop:
            self._guardar_log({
                "symbol": symbol,
                "trailing_pct": trailing_pct,
                "precio_entrada": entry,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }, tipo="trailing")

        try:
            _append_open_order({
                "symbol": symbol,
                "side": side.lower(),
                "precio_entrada": entry,
                "cantidad": qty,
                "take_profit": tp,
                "stop_loss": sl,
                "score": score,
                "timestamp": data["timestamp"]
            })
        except Exception as e:
            print(f"⚠️ No se pudo registrar en open_orders: {e}")

        # ===========================================================
        # 🖥️ Mostrar resumen visual en consola (modo simulado)
        # ===========================================================
        try:
            risk_usdt = self.capital_per_trade
            rr_ratio = abs((tp - entry) / (entry - sl)) if side.lower() == "buy" else abs((entry - sl) / (tp - entry))
            reward_usdt = risk_usdt * rr_ratio

            print_order_summary(
                symbol=symbol,
                side=side,
                entry=entry,
                tp=tp,
                sl=sl,
                qty=qty,
                score=score,
                conf=confianza,
                rr_ratio=rr_ratio,
                risk_usdt=risk_usdt,
                reward_usdt=reward_usdt,
                trailing_pct=trailing_pct
            )
        except Exception as e:
            print(f"⚠️ No se pudo mostrar resumen visual: {e}")


        return data

    # ===============================================================
    # 🧩 API EXTERNA: llamada desde signal_reevaluator
    # ===============================================================
    def colocar_orden_simulada(self, symbol, side, quantity, entry_price,
                               take_profit, stop_loss,
                               estrategia="N/A", direccion="neutro",
                               motivo="Simulación externa",
                               score_total=0.5, confluencia=0.0):
        """
        Registra una orden simulada externa con score técnico avanzado.
        """
        try:
            return self._simular_orden_avanzada(
                symbol=symbol,
                side=side,
                entry=entry_price,
                qty=quantity,
                sl=stop_loss,
                tp=take_profit,
                score=score_total,
                motivo=f"{motivo} | {estrategia} ({direccion})",
                confluencia=confluencia
            )
        except Exception as e:
            print(f"❌ Error en colocar_orden_simulada avanzada: {e}")
            return None

    # ===============================================================
    # ⚙️ ORDEN REAL (placeholder — mantiene compatibilidad)
    # ===============================================================
    def ejecutar_orden_con_sl_tp(self, side="buy", custom_symbol=None):
        """
        Mantiene compatibilidad con ejecución real (SPOT).
        """
        symbol = custom_symbol if custom_symbol else self.symbol
        print(f"⚙️ [Modo Avanzado] Ejecutar orden real en {symbol} ({side}) no implementado aún.")
        return self._simular_orden_avanzada(symbol, side, entry=100.0, score=0.75, motivo="Placeholder real")


# ===============================================================
# 🖥️ VISUALIZADOR DE ÓRDENES SIMULADAS (Rich Console)
# ---------------------------------------------------------------
# Muestra en consola los detalles de la orden ejecutada
# ===============================================================
console = Console()

def print_order_summary(symbol, side, entry, tp, sl, qty, score, conf, rr_ratio, risk_usdt, reward_usdt, trailing_pct):
    """
    Imprime una tabla en consola con los detalles completos de la orden simulada.
    """
    table = Table(title=f"📊 Simulación de Orden — {symbol} ({side.upper()})", style="bold cyan")
    table.add_column("Parámetro", justify="right", style="bold")
    table.add_column("Valor", justify="left")

    table.add_row("📈 Precio Entrada", f"{entry:.6f}")
    table.add_row("🎯 Take Profit", f"{tp:.6f}")
    table.add_row("🛑 Stop Loss", f"{sl:.6f}")
    table.add_row("💰 Cantidad", f"{qty:.2f}")
    table.add_row("⚖️ Riesgo (USDT)", f"{risk_usdt:.2f}")
    table.add_row("💵 Ganancia Potencial", f"{reward_usdt:.2f}")
    table.add_row("📊 R:R", f"{rr_ratio:.2f}")
    table.add_row("🧠 Score Técnico", f"{score:.2f}")
    table.add_row("📈 Confianza", f"{conf:.1f}%")
    table.add_row("🔁 Trailing Stop (%)", f"{trailing_pct * 100:.2f}%")

    console.print(table)