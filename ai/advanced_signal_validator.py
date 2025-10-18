# ai/advanced_signal_validator.py
# ================================================================
# 🧠 Validador avanzado de confluencias técnicas
# ------------------------------------------------
# Confirma LONG/SHORT sólo si hay coherencia entre:
# EMA, RSI, MACD, Volumen, Bollinger, ADX y patrón de velas.
# Soporta validación multi-timeframe y score ponderado.
# ================================================================

from __future__ import annotations
import math, sys, os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataclasses import dataclass
from typing import Dict, Any, Optional
from indicators.pattern_context_analyzer import analizar_contexto_velas
from ai.entry_timing_analyzer import evaluar_timing_entrada
from ai.entry_timing_analyzer_multilayer import evaluar_timing_multilayer


# ================================================================
# 📊 Utilidades técnicas básicas
# ================================================================
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = (up.rolling(n).mean()) / (down.rolling(n).mean())
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    line = ema_fast - ema_slow
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def adx(df: pd.DataFrame, n: int = 14):
    """Calcula ADX (Average Directional Index)."""
    if len(df) < n + 1:
        return pd.Series([0] * len(df))
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * (plus_dm.ewm(span=n).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=n).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_val = dx.ewm(span=n).mean()
    return adx_val

def bollinger(series: pd.Series, n=20, k=2.0):
    ma = series.rolling(n).mean()
    sd = series.rolling(n).std()
    upper, lower = ma + k*sd, ma - k*sd
    bbw = (upper - lower) / ma
    return upper, ma, lower, bbw

# ================================================================
# 🕯️ Patrones de velas
# ================================================================
def is_bullish_engulf(df: pd.DataFrame) -> bool:
    c1o, c1c = df["open"].iloc[-2], df["close"].iloc[-2]
    c2o, c2c = df["open"].iloc[-1], df["close"].iloc[-1]
    return (c1c < c1o) and (c2c > c2o) and (c2c >= c1o) and (c2o <= c1c)

def is_bearish_engulf(df: pd.DataFrame) -> bool:
    c1o, c1c = df["open"].iloc[-2], df["close"].iloc[-2]
    c2o, c2c = df["open"].iloc[-1], df["close"].iloc[-1]
    return (c1c > c1o) and (c2c < c2o) and (c2c <= c1o) and (c2o >= c1c)

def is_hammer(df: pd.DataFrame) -> bool:
    o, c, h, l = df["open"].iloc[-1], df["close"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1]
    body = abs(c - o); lower = min(o, c) - l; upper = h - max(o, c)
    return body > 0 and lower >= 2 * body and upper <= body

def is_shooting_star(df: pd.DataFrame) -> bool:
    o, c, h, l = df["open"].iloc[-1], df["close"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1]
    body = abs(c - o); upper = h - max(o, c); lower = min(o, c) - l
    return body > 0 and upper >= 2 * body and lower <= body

# ================================================================
# ⚙️ Configuración por defecto
# ================================================================
DEFAULTS = {
    "ema_fast": 12,
    "ema_slow": 26,
    "ema_trend": 200,
    "rsi_len": 14,
    "rsi_min_long": 50,
    "rsi_max_short": 50,
    "macd_hist_min": 0.001,
    "vol_window": 20,
    "vol_mult": 1.15,
    "bbw_min": 0.004,
    "adx_min": 20,
    "pattern_required": True,
    "htf_confirm": True,  # activa validación 15m
    "weights": {
        "ema": 0.20,
        "rsi": 0.15,
        "macd": 0.15,
        "vol": 0.10,
        "adx": 0.10,
        "pattern": 0.10,
        "contexto": 0.10,
        "timing": 0.10
    }
}

# ================================================================
# 🧮 Estructuras de resultado
# ================================================================

@dataclass
class CheckResult:
    ok: bool
    detail: str

@dataclass
class ValidationResult:
    ok: bool
    score: float
    direction: str
    checks: Dict[str, CheckResult]
    reasons: list

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "score": round(self.score, 3),
            "direction": self.direction,
            "checks": {k: {"ok": v.ok, "detail": v.detail} for k, v in self.checks.items()},
            "reasons": self.reasons
        }

# ================================================================
# 🔍 Validador principal
# ================================================================
def validar_confluencia(df_3m: pd.DataFrame, direction: str,
                        df_15m: Optional[pd.DataFrame] = None, cfg: Optional[dict] = None) -> dict:
    if df_3m is None or len(df_3m) < 60:
        return ValidationResult(False, 0.0, direction, {}, ["dataset insuficiente"]).to_dict()

    cfg = {**DEFAULTS, **(cfg or {})}
    direction = direction.lower()
    w = cfg["weights"]

    close = df_3m["close"].astype(float)
    ema_f = ema(close, cfg["ema_fast"])
    ema_s = ema(close, cfg["ema_slow"])
    ema_t = ema(close, cfg["ema_trend"])
    _rsi = rsi(close, cfg["rsi_len"])
    macd_line, macd_sig, macd_hist = macd(close)
    bb_u, bb_m, bb_l, bbw = bollinger(close)
    adx_val = adx(df_3m)
    vmean = df_3m["volume"].rolling(cfg["vol_window"]).mean()

    v_ok = df_3m["volume"].iloc[-1] >= cfg["vol_mult"] * float(vmean.iloc[-1] or 1)
    bbw_ok = float(bbw.iloc[-1] or 0) >= cfg["bbw_min"]
    adx_ok = float(adx_val.iloc[-1] or 0) >= cfg["adx_min"]

    ema_long_ok = (close.iloc[-1] > ema_f.iloc[-1] > ema_s.iloc[-1] > ema_t.iloc[-1])
    ema_short_ok = (close.iloc[-1] < ema_f.iloc[-1] < ema_s.iloc[-1] < ema_t.iloc[-1])
    rsi_long_ok = _rsi.iloc[-1] >= cfg["rsi_min_long"] and _rsi.iloc[-1] <= 70
    rsi_short_ok = _rsi.iloc[-1] <= cfg["rsi_max_short"] and _rsi.iloc[-1] >= 30
    macd_long_ok = macd_hist.iloc[-1] > cfg["macd_hist_min"] and macd_line.iloc[-1] > macd_sig.iloc[-1]
    macd_short_ok = macd_hist.iloc[-1] < -cfg["macd_hist_min"] and macd_line.iloc[-1] < macd_sig.iloc[-1]

    bull_pat = (df_3m.shape[0] >= 2) and (is_bullish_engulf(df_3m[-2:]) or is_hammer(df_3m))
    bear_pat = (df_3m.shape[0] >= 2) and (is_bearish_engulf(df_3m[-2:]) or is_shooting_star(df_3m))

    checks, reasons = {}, []

    # ============================================================
    # 🔍 Evaluaciones principales
    # ============================================================
    if direction == "long":
        checks["ema"] = CheckResult(ema_long_ok, f"EMA: {ema_f.iloc[-1]:.4f}>{ema_s.iloc[-1]:.4f}>{ema_t.iloc[-1]:.4f}")
        checks["rsi"] = CheckResult(rsi_long_ok, f"RSI: {_rsi.iloc[-1]:.2f}")
        checks["macd"] = CheckResult(macd_long_ok, f"MACD hist: {macd_hist.iloc[-1]:.5f}")
        checks["vol"] = CheckResult(v_ok, f"Vol: {df_3m['volume'].iloc[-1]:.0f} vs {vmean.iloc[-1]:.0f}")
        checks["adx"] = CheckResult(adx_ok, f"ADX: {adx_val.iloc[-1]:.2f}")
        checks["pattern"] = CheckResult(bull_pat, f"bullish={bull_pat}")
    else:
        checks["ema"] = CheckResult(ema_short_ok, f"EMA: {ema_f.iloc[-1]:.4f}<{ema_s.iloc[-1]:.4f}<{ema_t.iloc[-1]:.4f}")
        checks["rsi"] = CheckResult(rsi_short_ok, f"RSI: {_rsi.iloc[-1]:.2f}")
        checks["macd"] = CheckResult(macd_short_ok, f"MACD hist: {macd_hist.iloc[-1]:.5f}")
        checks["vol"] = CheckResult(v_ok, f"Vol: {df_3m['volume'].iloc[-1]:.0f} vs {vmean.iloc[-1]:.0f}")
        checks["adx"] = CheckResult(adx_ok, f"ADX: {adx_val.iloc[-1]:.2f}")
        checks["pattern"] = CheckResult(bear_pat, f"bearish={bear_pat}")


    # ============================================================
    # 🧭 CONTEXTO DE VELAS
    # ============================================================
    contexto = analizar_contexto_velas(df_3m)
    ctx_ok = not ((contexto["tendencia"] == "bajista" and direction == "long") or
                  (contexto["tendencia"] == "alcista" and direction == "short"))
    checks["contexto"] = CheckResult(ctx_ok, f"{contexto['tendencia']} | score={contexto['trend_score']}")
    if not ctx_ok:
        reasons.append(f"❌ Contexto contrario — {contexto['detalle']}")

    # ============================================================
    # 🕒 TIMING DE ENTRADA (3m + 5m + 15m)
    # ============================================================
    timing_ok = True
    timing_detail = "N/A"
    modo_rebote = False

    if cfg["htf_confirm"] and df_15m is not None:
        # Si el modo multilayer está activado
        if cfg.get("timing_mode", "basic") == "multilayer":
            # ⚙️ Necesita también df_5m
            from data.data_loader import DataLoader
            try:
                data_loader = DataLoader(None)
                df_5m = data_loader.obtener_ohlcv("BTCUSDT", interval="5m", limit=300)
            except Exception:
                df_5m = None

            timing = evaluar_timing_multilayer(df_3m, df_5m, df_15m, direction)
        else:
            # Usa el analizador clásico 3m + 15m
            timing = evaluar_timing_entrada(df_3m, df_15m, direction)

        timing_ok = timing["valido"]
        modo_rebote = timing.get("modo_rebote", False)

        if cfg.get("timing_mode", "basic") == "multilayer":
            timing_detail = (
                f"{timing['razon']} | etapa={timing['etapa']} | "
                f"3m={timing['tendencia_3m']} | 5m={timing['tendencia_5m']} | "
                f"15m={timing['tendencia_15m']} | delay≈{timing['delay_estimado']}m"
            )
        else:
            timing_detail = (
                f"{timing['razon']} | etapa={timing['etapa']} | "
                f"15m={timing['tendencia_15m']}"
            )

        checks["timing"] = CheckResult(timing_ok, timing_detail)
        if not timing_ok and not modo_rebote:
            reasons.append(f"⚠️ Timing débil — {timing_detail}")
    else:
        checks["timing"] = CheckResult(True, "sin validación 15m")

    # ============================================================
    # 🧮 Cálculo de score técnico total
    # ============================================================
    score = sum(w.get(k, 0) * (1.0 if v.ok else 0.0) for k, v in checks.items())
    ok = score >= 0.65 or modo_rebote

    for k, v in checks.items():
        if not v.ok:
            reasons.append(f"{k} falla → {v.detail}")

    if modo_rebote:
        reasons.append("⚡ Rebote técnico detectado — operación de reversión controlada.")

    return ValidationResult(ok, float(score), direction, checks, reasons).to_dict()

# ======================================================
# 🎨 VISUALIZADOR DE VALIDACIÓN AVANZADA
# ======================================================
def mostrar_validacion_visual(df_3m: pd.DataFrame, direction: str = "long",
                              df_15m: Optional[pd.DataFrame] = None, cfg: dict = None):
    """
    Ejecuta validar_confluencia() y muestra los resultados con colores y emojis.
    Incluye evaluación visual del timing (3m+15m) y contexto multi-timeframe.
    """
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    console.print("\n🤖 [bold cyan]VALIDACIÓN DE CONFLUENCIAS TÉCNICAS AVANZADA[/bold cyan]\n")

    resultado = validar_confluencia(df_3m, direction, df_15m, cfg or {})
    checks = resultado.get("checks", {})
    score = resultado.get("score", 0)
    ok = resultado.get("ok", False)
    reasons = resultado.get("reasons", [])
    direction = resultado.get("direction", "N/A")

    # ======================================================
    # TABLA PRINCIPAL DE INDICADORES
    # ======================================================
    table = Table(
        box=box.ROUNDED,
        title=f"📈 Dirección: {direction.upper()} — Score: {score:.2f}",
        title_style="bold cyan"
    )
    table.add_column("Indicador", justify="left", style="bold")
    table.add_column("Detalle", justify="center")
    table.add_column("Resultado", justify="center")

    for k, v in checks.items():
        symbol = "✅" if v["ok"] else "❌"
        color = "green" if v["ok"] else "red"

        # 🎯 Estilo especial para Timing
        if k == "timing":
            if "reversión" in v["detail"] or "rebote" in v["detail"]:
                color, symbol = "cyan", "⚡"
            elif "contratendencia" in v["detail"]:
                color, symbol = "red", "🚫"
            elif "inicio" in v["detail"]:
                color, symbol = "green", "🟢"
            elif "extendido" in v["detail"]:
                color, symbol = "yellow", "⚠️"
            elif "final" in v["detail"]:
                color, symbol = "red", "⛔"

        # 🎯 Estilo especial para Contexto
        if k == "contexto":
            if "bajista" in v["detail"]:
                color, symbol = "red", "📉"
            elif "alcista" in v["detail"]:
                color, symbol = "green", "📈"
            elif "neutro" in v["detail"]:
                color, symbol = "yellow", "⚖️"

        table.add_row(k.upper(), v["detail"], f"[{color}]{symbol}[/{color}]")

    console.print(table)

    # ======================================================
    # RESUMEN FINAL DE CONFLUENCIA
    # ======================================================
    if ok:
        if any("⚡" in r for r in reasons):
            console.print("\n[cyan bold]⚡ Rebote técnico válido — entrada anticipada de reversión controlada.[/cyan bold]")
        else:
            console.print(f"\n[green bold]✅ Señal confirmada ({direction.upper()})[/green bold] — Confluencias técnicas coherentes y timing favorable.")
    else:
        console.print(f"\n[red bold]❌ Sin confluencia suficiente[/red bold] — Razones:")
        for r in reasons:
            color = "yellow" if "⚠️" in r else "red"
            console.print(f"   [{color}]{r}[/{color}]")

    # ======================================================
    # INFO EXTRA SI EXISTE TIMING
    # ======================================================
    if "timing" in checks:
        detalle = checks["timing"]["detail"]
        if "inicio" in detalle:
            console.print("\n🟢 [bold green]Entrada temprana detectada — excelente momento.[/bold green]")
        elif "extendido" in detalle:
            console.print("\n⚠️ [bold yellow]Entrada extendida — posible retroceso cercano.[/bold yellow]")
        elif "final" in detalle:
            console.print("\n🔴 [bold red]Entrada tardía — movimiento agotado.[/bold red]")
        elif "contratendencia" in detalle:
            console.print("\n🚫 [bold red]Contratendencia — riesgo elevado de reversión.[/bold red]")

    console.print("\n📊 [bold blue]Validación técnica completada.[/bold blue]\n")
# ======================================================
# 🚀 EJECUCIÓN DIRECTA PARA PRUEBA / DEPURACIÓN
# ======================================================
if __name__ == "__main__":
    import sys, os, json, time
    from binance.client import Client
    from data.data_loader import DataLoader
    from rich.console import Console

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    console = Console()
    console.print("\n🔍 [bold cyan]Iniciando prueba del validador avanzado con multi-timeframe (3m + 15m)...[/bold cyan]\n")

    # ======================================================
    # 🔧 Cargar configuración
    # ======================================================
    try:
        with open("config/settings.json") as f:
            config = json.load(f)
        client = Client(config["api_key"], config["api_secret"])
        data_loader = DataLoader(client)
    except Exception as e:
        console.print(f"[red]❌ Error al cargar configuración o cliente Binance: {e}[/red]")
        sys.exit(1)

    # ======================================================
    # 🧩  Parámetros de prueba
    # ======================================================
    symbol = config.get("default_symbol", "BTCUSDT")
    timing_mode = config.get("timing_mode", "multilayer")  # puede ser "basic" o "multilayer"
    direction = config.get("test_direction", "long")       # para probar señales long o short

    console.print(f"📊 Descargando datos de {symbol} (3m, 5m, 15m)...")
    df_3m = data_loader.obtener_ohlcv(symbol, interval="3m", limit=300)
    df_5m = data_loader.obtener_ohlcv(symbol, interval="5m", limit=300) if timing_mode == "multilayer" else None
    df_15m = data_loader.obtener_ohlcv(symbol, interval="15m", limit=300)
    time.sleep(1)

    if df_3m is None or df_3m.empty or df_15m is None or df_15m.empty:
        console.print("[red]❌ No se pudieron obtener datos suficientes desde Binance.[/red]")
        sys.exit(1)

    # ======================================================
    # 🧠 Ejecutar validación avanzada
    # ======================================================
    console.print("\n⚙️ [bold blue]Ejecutando validación técnica completa...[/bold blue]")
    mostrar_validacion_visual(
        df_3m=df_3m,
        direction=direction,
        df_15m=df_15m,
        cfg={
            "htf_confirm": True,
            "pattern_required": True,
            "timing_mode": timing_mode  # activa multilayer dinámicamente
        }
    )

    # ======================================================
    # 🔍 Informe final
    # ======================================================
    modo = "Multi-layer (3m+5m+15m)" if timing_mode == "multilayer" else "Básico (3m+15m)"
    console.print(f"\n✅ [bold green]Prueba de validación avanzada finalizada correctamente.[/bold green]")
    console.print(f"🧩 Modo de análisis: [cyan]{modo}[/cyan]\n")

