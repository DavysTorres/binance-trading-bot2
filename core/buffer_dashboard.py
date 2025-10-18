# core/buffer_dashboard.py
# ===============================================================
# 📊 Buffer Dashboard — Visualizador de señales activas
# ---------------------------------------------------------------
# Muestra el estado actual del buffer (nuevas, pendientes,
# confirmadas, ejecutadas, canceladas) en formato Rich Table.
# Permite refrescar dinámicamente y filtrar por estado.
# ===============================================================

import os, json, time
from rich.console import Console
from rich.table import Table
from rich import box
from datetime import datetime

# Config
BUFFER_PATH = "signals_validated.json"
REFRESH_INTERVAL = 10  # segundos entre actualizaciones automáticas

console = Console()


# ===============================================================
# 🧩 Utilidad: cargar buffer
# ===============================================================
def load_buffer():
    if not os.path.exists(BUFFER_PATH):
        return []
    try:
        with open(BUFFER_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


# ===============================================================
# 🎨 Colores por estado
# ===============================================================
STATE_COLORS = {
    "nueva": "bright_cyan",
    "pendiente": "yellow",
    "confirmada": "green",
    "ejecutada": "bright_blue",
    "cancelada": "red",
}


# ===============================================================
# 📊 Renderizar tabla
# ===============================================================
def render_dashboard(data, filtro_estado=None):
    console.clear()
    console.rule(f"[bold cyan]📊 DASHBOARD DE SEÑALES — {datetime.now().strftime('%H:%M:%S')}[/bold cyan]")

    if filtro_estado:
        data = [s for s in data if s.get("estado") == filtro_estado]

    if not data:
        console.print("[bold red]⚠️ No hay señales registradas actualmente.[/bold red]")
        return

    # Ordenar por estado y timestamp
    data = sorted(data, key=lambda x: (x.get("estado", ""), x.get("timestamp", "")))

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("Símbolo", justify="left", style="bold cyan")
    table.add_column("Tipo", justify="center")
    table.add_column("Estrategia", justify="center")
    table.add_column("Estado", justify="center")
    table.add_column("Etapa", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Intentos", justify="center")
    table.add_column("Timestamp", justify="center")
    table.add_column("Detalle", justify="left")

    for s in data:
        estado = s.get("estado", "desconocido")
        color = STATE_COLORS.get(estado, "white")

        table.add_row(
            s.get("symbol", "N/A"),
            s.get("tipo", "N/A").upper(),
            s.get("estrategia", "-"),
            f"[{color}]{estado.upper()}[/{color}]",
            s.get("etapa", "-"),
            str(round(float(s.get("score_total", 0)), 2)),
            str(s.get("intentos", 0)),
            s.get("timestamp", ""),
            s.get("razon", s.get("reversa_ult2", "-")),
        )

    console.print(table)


# ===============================================================
# 🌀 Modo interactivo (refresco automático)
# ===============================================================
def run_dashboard(filtro_estado=None, auto_refresh=True):
    try:
        while True:
            data = load_buffer()
            render_dashboard(data, filtro_estado)
            if not auto_refresh:
                break
            console.print(f"\n🔄 Refrescando en {REFRESH_INTERVAL} segundos...", style="dim")
            time.sleep(REFRESH_INTERVAL)
    except KeyboardInterrupt:
        console.print("\n👋 [bold cyan]Dashboard detenido por el usuario.[/bold cyan]")


# ===============================================================
# 🚀 Ejecución directa
# ===============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualizador de señales activas del buffer.")
    parser.add_argument("--estado", help="Filtrar por estado (nueva, pendiente, confirmada, ejecutada, cancelada)", default=None)
    parser.add_argument("--once", action="store_true", help="Muestra una sola vez sin refresco automático")
    args = parser.parse_args()

    run_dashboard(filtro_estado=args.estado, auto_refresh=not args.once)
