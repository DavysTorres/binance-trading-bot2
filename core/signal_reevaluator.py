# core/signal_reevaluator.py

# ================================================================
# 🔁 Signal Reevaluator (versión inteligente con multilayer)
# ---------------------------------------------------------------
# Reevaluación dinámica de señales con ventana de 2 velas (≈6 min)
#   ✅ Verifica reversas y sincronía 3m/5m/15m.
#   ✅ Confirma o cancela antes de ejecutar orden.
#   ✅ Integra risk/reward automático.
# ================================================================

import os, json, sys, time, math
import pandas as pd
import numpy as np
from datetime import datetime
from binance.client import Client
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ---------------------------------------------------------------
# 📦 Imports del proyecto
# ---------------------------------------------------------------
from data.data_loader import DataLoader
from ai.entry_timing_analyzer_multilayer import evaluar_timing_multilayer
from ai.advanced_signal_validator import validar_confluencia
from core.signal_buffer import SignalBuffer
from trading.order_manager import OrderManager
from core.notification_manager import send_info
from core.report_manager import registrar_evento, generar_reporte_diario, enviar_resumen_console

# ---------------------------------------------------------------
# 🔧 Configuración dinámica
# ---------------------------------------------------------------
with open("config/settings.json") as f:
    config = json.load(f)

API_KEY = config["api_key"]
API_SECRET = config["api_secret"]
client = Client(API_KEY, API_SECRET)
data_loader = DataLoader(client)
order_manager = OrderManager()
buffer = SignalBuffer(config.get("signals_validated_path", "signals_validated.json"))
console = Console()

LOOP_INTERVAL = config.get("loop_interval_seconds", 180)
MAX_EXECUTIONS = config.get("max_executions_per_cycle", 3)
SCORE_THRESHOLD = config.get("score_threshold", 0.55)
CONFLUENCE_THRESHOLD = config.get("confluence_threshold", 0.60)
RISK_REWARD = config.get("risk_reward_ratio", 1.5)
RISK_PER_TRADE_USDT = config.get("risk_per_trade_usdt", 5)

# ================================================================
# ⚙️ Utilidades
# ================================================================
def _get_symbol_filters(symbol):
    try:
        info = client.get_symbol_info(symbol)
        for f in info["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                tick_size = float(f["tickSize"])
            if f["filterType"] == "LOT_SIZE":
                step_size = float(f["stepSize"])
                min_qty = float(f["minQty"])
            if f["filterType"] == "MIN_NOTIONAL":
                min_notional = float(f["minNotional"])
        return tick_size, step_size, min_qty, min_notional
    except Exception:
        return 0.01, 0.0001, 0.0001, 10.0


def _calc_atr(df, period=14):
    if len(df) < period + 1:
        return 0.0
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    return float(atr or 0)


def _compute_tp_sl(entry, direction, atr):
    entry = float(entry)
    base_sl = atr * 1.2
    if direction == "long":
        sl = entry - base_sl
        tp = entry + base_sl * RISK_REWARD
    else:
        sl = entry + base_sl
        tp = entry - base_sl * RISK_REWARD
    return round(tp, 6), round(sl, 6)


def _position_size(entry, sl, step_size, min_qty, min_notional, risk_usdt=RISK_PER_TRADE_USDT):
    stop_dist = abs(entry - sl)
    qty = risk_usdt / stop_dist if stop_dist > 0 else 0
    qty = math.floor(qty / step_size) * step_size
    if entry * qty < min_notional:
        qty = math.floor(min_notional / entry / step_size) * step_size
    return max(qty, min_qty)

# ================================================================
# 🧩 Visualización Rich
# ================================================================
def _mostrar_tabla(result, symbol):
    table = Table(box=box.ROUNDED, title=f"📊 Reevaluación {symbol}", title_style="bold cyan")
    for k, v in result.items():
        table.add_row(k, str(v))
    console.print(table)

# ================================================================
# 🔁 Reevaluación avanzada con multilayer
# ================================================================
def reevaluar_senales():
    señales = buffer.list_fresh()
    if not señales:
        console.print("⚠️ No hay señales para reevaluar.")
        return

    ejecutadas = 0
    buffer.clean_expired()

    for s in señales:
        symbol = s.get("symbol")
        direction = s.get("tipo")
        score_prev = s.get("score_total", 0)
        t_inicio = s.get("timestamp")

        console.print(f"\n🔍 Reevaluando [bold yellow]{symbol}[/bold yellow] — {direction.upper()} ...")

        # === 1️⃣ Descarga de datos multi-timeframe ===
        df3 = data_loader.obtener_ohlcv(symbol, interval="3m", limit=150)
        df5 = data_loader.obtener_ohlcv(symbol, interval="5m", limit=150)
        df15 = data_loader.obtener_ohlcv(symbol, interval="15m", limit=150)
        if df3 is None or df5 is None or df15 is None:
            console.print(f"⚠️ Sin datos suficientes para {symbol}")
            buffer.increment_attempts(symbol)
            continue

        # === 2️⃣ Evaluación multilayer ===
        result = evaluar_timing_multilayer(df3, df5, df15, direction)
        _mostrar_tabla(result, symbol)

        # Si está en etapa de pullback o temprano, esperar 2 velas (~6min)
        if result["etapa"] in ("pullback", "temprano") or result["pullback_saludable"]:
            console.print(f"⏳ Esperando 2 velas para confirmar señal de {symbol}...")
            buffer.mark_pending(symbol, result)
            continue

        # === 3️⃣ Detección de reversa ===
        reversa = result.get("reversa_ult2", "none")
        if (direction == "long" and reversa == "bearish") or (direction == "short" and reversa == "bullish"):
            console.print(f"🚫 Cancelada — patrón de reversa detectado en {symbol} ({reversa}).")
            buffer.cancel(symbol, reason=f"Reversa detectada ({reversa})")
            continue

        # === 4️⃣ Confirmación de confluencias ===
        val = validar_confluencia(df3, direction, df15)
        if not val["ok"] or val["score"] < CONFLUENCE_THRESHOLD:
            console.print(f"⚠️ Sin confluencia técnica suficiente ({val['score']:.2f}). Reevaluar en próxima vela.")
            buffer.increment_attempts(symbol)
            continue

        # === 5️⃣ Si todo se confirma, ejecutar ===
        atr = _calc_atr(df3)
        entry = float(df3["close"].iloc[-1])
        tp, sl = _compute_tp_sl(entry, direction, atr)
        tick, step, min_qty, min_notional = _get_symbol_filters(symbol)
        qty = _position_size(entry, sl, step, min_qty, min_notional)

        if qty <= 0:
            console.print(f"⚠️ No se pudo calcular tamaño de posición válido ({symbol}).")
            continue

        console.print(f"✅ [green]Confirmada — Ejecutando {direction.upper()} en {symbol}[/green]")
        orden = order_manager.colocar_orden_simulada(
            symbol=symbol,
            side=("BUY" if direction == "long" else "SELL"),
            quantity=qty,
            entry_price=entry,
            take_profit=tp,
            stop_loss=sl,
            estrategia="multilayer",
            direccion=direction,
            motivo=f"Confirmación multilayer {result['etapa']} | score={result['score']:.2f}",
            score_total=result["score"],
            confluencia=val["score"]
        )

        registrar_evento(symbol, "simulada", "confirmada", True)
        send_info("Sistema", f"✅ Señal confirmada {symbol} ({direction.upper()}) | Score={result['score']:.2f}")
        ejecutadas += 1

        if ejecutadas >= MAX_EXECUTIONS:
            console.print("🔒 Límite de ejecuciones alcanzado.")
            break

        buffer.remove(symbol)
        time.sleep(1.5)

# ================================================================
# ⏱️ Loop principal
# ================================================================
def loop_reevaluador():
    while True:
        console.print(f"\n⏱️ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Ciclo de reevaluación iniciado...")
        reevaluar_senales()
        console.print(f"🛌 Esperando {LOOP_INTERVAL} segundos antes del próximo ciclo...\n")
        time.sleep(LOOP_INTERVAL)
        reporte = generar_reporte_diario()
        enviar_resumen_console(reporte)

# ================================================================
# 🚀 Ejecución directa
# ================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reevaluador técnico multilayer (3m/5m/15m + reversas).")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.loop:
        loop_reevaluador()
    elif args.once:
        reevaluar_senales()
    else:
        print("❗ Usa --loop o --once para ejecutar correctamente.")
