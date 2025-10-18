# indicators/signal_analyzer.py

import sys, os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.market_scanner_module import MarketScanner

# ======================
# 🧩 Imports dinámicos
# ======================
try:
    from .rsi import calcular_rsi
    from .ema import calcular_ema
    from .volumen import calcular_volumen_promedio, es_breakout_volumen
    from .patterns import detectar_patron_general
    from .confirmation import aplicar_indicadores_confirmacion, calcular_confianza_tecnica
except ImportError:
    from indicators.rsi import calcular_rsi
    from indicators.ema import calcular_ema
    from indicators.volumen import calcular_volumen_promedio, es_breakout_volumen
    from indicators.patterns import detectar_patron_general
    from indicators.confirmation import aplicar_indicadores_confirmacion, calcular_confianza_tecnica


# ======================================================
# 🔧 Función de protección para EMAs
# ======================================================
def asegurar_emas(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "EMA_fast" not in df.columns:
        df["EMA_fast"] = df["close"].ewm(span=12, adjust=False).mean()
    if "EMA_slow" not in df.columns:
        df["EMA_slow"] = df["close"].ewm(span=26, adjust=False).mean()
    df["EMA_fast"] = df["EMA_fast"].bfill()
    df["EMA_slow"] = df["EMA_slow"].bfill()
    return df


# ======================================================
# 🧠 Motor de análisis técnico avanzado
# ======================================================
def detectar_senal(df: pd.DataFrame) -> dict:
    """
    Analizador técnico avanzado con cálculo de score ponderado.
    Combina EMAs, MACD, RSI, ADX, StochRSI, Volumen, Volatilidad y Patrón.
    """

    resultado = {"entrada": False, "estrategia": None, "tipo": "neutro", "detalles": {}}

    if df is None or df.empty or len(df) < 50:
        return {"entrada": False, "estrategia": "Datos insuficientes", "tipo": "neutro"}

    # 🔧 Columnas esenciales
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = 0.0
    # 🔧 Convertir solo columnas numéricas a float
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # mantener columnas datetime u objetos como están
            continue
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 🧭 Sesgo global
    try:
        scanner = MarketScanner()
        sesgo = scanner.detectar_sesgo_mercado()
        intensidad = scanner.calcular_intensidad_mercado()
    except Exception:
        sesgo, intensidad = "neutral", 0.5

    # ============================
    # 📊 Cálculo de indicadores
    # ============================
    df = calcular_rsi(df)
    df = calcular_ema(df)
    df = aplicar_indicadores_confirmacion(df)
    df["volumen_movil_20"] = calcular_volumen_promedio(df, 20)
    df = asegurar_emas(df)

    # Bandas de Bollinger y ATR
    df["MA20"] = df["close"].rolling(20).mean()
    df["STD20"] = df["close"].rolling(20).std()
    df["BB_upper"] = df["MA20"] + 2 * df["STD20"]
    df["BB_lower"] = df["MA20"] - 2 * df["STD20"]
    df["BBW"] = ((df["BB_upper"] - df["BB_lower"]) / df["MA20"]) * 100
    df["TR"] = (df[["high", "close"]].max(axis=1) - df[["low", "close"]].min(axis=1))
    df["ATR_14"] = df["TR"].rolling(14).mean()

    ultima = df.iloc[-1]
    penultima = df.iloc[-2]

    # ============================
    # ⚙️ Ajustes según sesgo
    # ============================
    if sesgo == "bullish":
        rsi_overbought = 75
        rsi_oversold = 25
    elif sesgo == "bearish":
        rsi_overbought = 65
        rsi_oversold = 35
    else:
        rsi_overbought = 70
        rsi_oversold = 30

    # ============================
    # 📈 Indicadores individuales
    # ============================
    ema_fast, ema_slow = ultima["EMA_fast"], ultima["EMA_slow"]
    macd = ultima.get("MACD", 0)
    signal_macd = ultima.get("Signal_MACD", 0)
    adx = ultima.get("ADX", 20)
    rsi_val = ultima.get("RSI_14", 50)
    stoch_rsi = ultima.get("StochRSI", 50)
    volumen = ultima.get("volume", 0)
    vol_prom = ultima.get("volumen_movil_20", 1)
    bbw = ultima.get("BBW", 1)
    atr_rel = ultima.get("ATR_14", 0) / ultima.get("close", 1)
    patron = detectar_patron_general(df)

    # ============================
    # 💡 Cálculo del score técnico
    # ============================
    pesos = {
        "ema": 0.25,
        "rsi": 0.15,
        "macd": 0.15,
        "adx": 0.10,
        "stoch": 0.10,
        "volumen": 0.10,
        "volatilidad": 0.10,
        "patron": 0.05
    }

    scores = {}

    # EMA
    if ema_fast > ema_slow:
        scores["ema"] = 1.0
    elif ema_fast < ema_slow:
        scores["ema"] = -1.0
    else:
        scores["ema"] = 0

    # RSI
    if 40 < rsi_val < 60:
        scores["rsi"] = 0.2
    elif rsi_val >= 60:
        scores["rsi"] = 1
    elif rsi_val <= 40:
        scores["rsi"] = -1

    # MACD
    scores["macd"] = 1 if macd > signal_macd else -1

    # ADX
    scores["adx"] = 1 if adx >= 25 else 0

    # StochRSI
    if stoch_rsi > 80:
        scores["stoch"] = -0.5
    elif stoch_rsi < 20:
        scores["stoch"] = 0.5
    else:
        scores["stoch"] = 0

    # Volumen
    scores["volumen"] = 1 if volumen > vol_prom * 1.2 else 0

    # Volatilidad
    scores["volatilidad"] = 1 if bbw > 1.5 or atr_rel > 0.004 else 0

    # Patrón
    scores["patron"] = 1 if patron not in ["none", None] else 0

    # Calcular score total
    score_total = sum(scores[k] * pesos[k] for k in pesos)
    score_total = max(min(score_total, 1), -1)  # limitar entre -1 y 1

    # ============================
    # 🎯 Interpretación
    # ============================
    tipo = "long" if score_total > 0.25 else "short" if score_total < -0.25 else "neutro"

    estrategia = (
        "Momentum Alcista Avanzado" if tipo == "long" else
        "Momentum Bajista Avanzado" if tipo == "short" else
        "Sin Señal Clara / Consolidación"
    )

    # ============================
    # 🚀 Resultado final
    # ============================
    resultado = {
        "entrada": tipo in ["long", "short"],
        "estrategia": estrategia,
        "tipo": tipo,
        "score_total": round(score_total, 3),
        "sesgo": sesgo,
        "intensidad": intensidad,
        "detalles": {
            **scores,
            "EMA_fast": round(ema_fast, 6),
            "EMA_slow": round(ema_slow, 6),
            "RSI_14": round(rsi_val, 2),
            "MACD": round(macd, 6),
            "Signal_MACD": round(signal_macd, 6),
            "ADX": round(adx, 2),
            "StochRSI": round(stoch_rsi, 2),
            "volumen": round(volumen, 2),
            "volumen_promedio": round(vol_prom, 2),
            "BBW_%": round(bbw, 3),
            "ATR_rel": round(atr_rel, 4),
            "patron_vela": patron,
            "score_total": round(score_total, 3),
        },
    }

    return resultado

# ======================================================
# 🎨 VISUALIZADOR DE RESULTADOS EN CONSOLA
# ======================================================
def mostrar_resumen_visual(resultado: dict):
    """
    Muestra el análisis técnico con colores y emojis.
    Requiere librería 'rich' → pip install rich
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        print("⚠️ Librería 'rich' no encontrada. Instálala con: pip install rich")
        print(json.dumps(resultado, indent=4))
        return

    console = Console()
    detalles = resultado.get("detalles", {})
    tipo = resultado.get("tipo", "neutro")
    estrategia = resultado.get("estrategia", "Desconocida")
    score = resultado.get("score_total", 0)
    sesgo = resultado.get("sesgo", "neutral")
    intensidad = resultado.get("intensidad", 0)

    console.print("\n🎯 [bold cyan]RESUMEN DE ANÁLISIS TÉCNICO[/bold cyan]\n")

    # Encabezado general
    resumen = f"[bold]{estrategia}[/bold]\n\n"
    resumen += f"📊 Tipo: [green]{tipo.upper()}[/green] | "
    resumen += f"🌍 Sesgo Global: [bold magenta]{sesgo.upper()}[/bold magenta] | "
    resumen += f"🔥 Intensidad: [yellow]{intensidad:.2f}[/yellow] | "
    resumen += f"🎯 Score: [bold]{score:.3f}[/bold]"

    console.print(resumen + "\n")

    # Crear tabla visual
    table = Table(box=box.ROUNDED, title="Indicadores Técnicos", title_style="bold cyan")
    table.add_column("Indicador", justify="left", style="bold")
    table.add_column("Valor", justify="center")
    table.add_column("Evaluación", justify="center")

    def eval_color(valor, ok_cond, neutral_cond=None):
        if ok_cond:
            return "[green]✅[/green]"
        elif neutral_cond:
            return "[yellow]⚠️[/yellow]"
        else:
            return "[red]❌[/red]"

    # RSI
    rsi = detalles.get("RSI_14", 0)
    if 40 <= rsi <= 60:
        rsi_color = eval_color(rsi, False, True)
    elif rsi > 60:
        rsi_color = eval_color(rsi, True)
    else:
        rsi_color = eval_color(rsi, False)

    # MACD
    macd = detalles.get("MACD", 0)
    signal_macd = detalles.get("Signal_MACD", 0)
    macd_color = eval_color(macd, macd > signal_macd)

    # ADX
    adx = detalles.get("ADX", 0)
    adx_color = eval_color(adx, adx >= 25)

    # StochRSI
    stoch = detalles.get("StochRSI", 50)
    if 20 <= stoch <= 80:
        stoch_color = eval_color(stoch, False, True)
    elif stoch > 80 or stoch < 20:
        stoch_color = eval_color(stoch, True)
    else:
        stoch_color = eval_color(stoch, False)

    # Volumen
    vol = detalles.get("volumen", 0)
    vmean = detalles.get("volumen_promedio", 1)
    vol_color = eval_color(vol, vol >= vmean * 1.2, vol >= vmean * 0.8)

    # Volatilidad
    bbw = detalles.get("BBW_%", 0)
    atr_rel = detalles.get("ATR_rel", 0)
    volat_color = eval_color(bbw, bbw > 1.5 or atr_rel > 0.004, bbw > 1.0)

    # Patrón de vela
    patron = detalles.get("patron_vela", "none")
    patron_color = "[green]✅[/green]" if patron not in ["none", None] else "[red]❌[/red]"

    # EMA
    ema_fast = detalles.get("EMA_fast", 0)
    ema_slow = detalles.get("EMA_slow", 0)
    ema_color = eval_color(ema_fast, ema_fast > ema_slow)

    # Añadir filas
    table.add_row("EMA", f"{ema_fast:.2f} / {ema_slow:.2f}", ema_color)
    table.add_row("RSI", f"{rsi:.2f}", rsi_color)
    table.add_row("MACD", f"{macd:.4f} / {signal_macd:.4f}", macd_color)
    table.add_row("ADX", f"{adx:.2f}", adx_color)
    table.add_row("StochRSI", f"{stoch:.2f}", stoch_color)
    table.add_row("Volumen", f"{vol:.2f} / {vmean:.2f}", vol_color)
    table.add_row("Volatilidad", f"BBW={bbw:.3f} | ATR%={atr_rel*100:.2f}", volat_color)
    table.add_row("Patrón Vela", patron, patron_color)

    console.print(table)
    console.print("\n📈 [bold blue]Análisis técnico completado.[/bold blue]\n")


# ======================================================
# 🚀 EJECUCIÓN DIRECTA PARA PRUEBA / DEPURACIÓN
# ======================================================
if __name__ == "__main__":
    from binance.client import Client
    from data.data_loader import DataLoader

    print("\n🔍 Iniciando prueba del analizador de señales...")
    try:
        # cargar configuración
        import json
        with open("config/settings.json") as f:
            config = json.load(f)

        symbol = config.get("default_symbol", "BTCUSDT")
        interval = config.get("ohlcv_interval", "3m")
        limit = config.get("ohlcv_limit", 150)

        # inicializar cliente Binance
        client = Client(config["api_key"], config["api_secret"])
        loader = DataLoader(client)
        df = loader.obtener_ohlcv(symbol, interval=interval, limit=limit)

        if df is None or df.empty:
            print(f"⚠️ No se pudieron obtener datos OHLCV para {symbol}")
        else:
            resultado = detectar_senal(df)
            print("\n📊 Resultado del análisis técnico:")
            mostrar_resumen_visual(resultado)

    except Exception as e:
        print(f"❌ Error al ejecutar prueba del analizador: {e}")
