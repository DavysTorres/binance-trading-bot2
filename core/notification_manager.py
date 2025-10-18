# core/notification_manager.py
import os
import json
import requests
from datetime import datetime
from colorama import Fore, Style, init

# Inicializar colorama para consola
init(autoreset=True)

# Cargar configuración
with open("config/settings.json", "r") as f:
    config = json.load(f)

TELEGRAM_TOKEN = config.get("telegram_bot_token")
TELEGRAM_CHAT_ID = config.get("telegram_chat_id")

LOG_PATH = "logs/notifications_log.json"
os.makedirs("logs", exist_ok=True)

# ----------------------------
# 🧩 Funciones auxiliares
# ----------------------------

def _guardar_log(evento):
    """Guarda los mensajes críticos en un log JSON."""
    try:
        logs = []
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r") as f:
                logs = json.load(f)
        logs.append(evento)
        with open(LOG_PATH, "w") as f:
            json.dump(logs[-100:], f, indent=4)  # solo últimos 100
    except Exception as e:
        print(f"{Fore.RED}⚠️ Error guardando log de notificación: {e}")

def send_telegram_message(text, parse_mode="Markdown"):
    """Envía mensaje a Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"{Fore.YELLOW}⚠️ Bot de Telegram no configurado correctamente en settings.json")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"{Fore.RED}❌ Error enviando mensaje a Telegram: {e}")

# ----------------------------
# 🎯 Formateo de mensajes
# ----------------------------

def format_message(symbol, message=None, tipo=None, estrategia=None,
                   precio_entrada=None, precio_salida=None, variacion=None,
                   duracion=None, tendencia=None, motivo=None, evento_sistema=False):
    """
    Crea un mensaje en Markdown para Telegram.
    Si evento_sistema=True → formato simplificado.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Mensaje de sistema (INFO, CRITICAL, etc.)
    if evento_sistema or not estrategia:
        return (
            f"🧠 *{symbol}*\n"
            f"🕒 {timestamp}\n\n"
            f"{message or motivo or 'Evento del sistema.'}"
        )

    # Mensaje de trading
    msg = f"📈 *Evento de Trading en {symbol}*\n"
    msg += f"🧭 Estrategia: *{estrategia or 'N/A'}*\n"
    msg += f"💹 Tipo: *{tipo.upper() if tipo else 'N/A'}*\n"

    if tendencia:
        emoji = "📊" if tendencia == "lateral" else ("🟢" if tendencia == "alcista" else "🔴")
        msg += f"{emoji} Tendencia: *{tendencia.capitalize()}*\n"

    if precio_entrada:
        msg += f"💰 Entrada: `{precio_entrada}`\n"
    if precio_salida:
        msg += f"🏁 Salida: `{precio_salida}`\n"
    if variacion is not None:
        color = "🟢" if variacion > 0 else "🔴"
        msg += f"{color} Variación: *{variacion:.2f}%*\n"
    if duracion:
        msg += f"⏱️ Duración: `{duracion}`\n"
    if motivo:
        msg += f"🧠 Motivo: _{motivo}_\n"

    msg += f"\n🕒 {timestamp}"
    return msg

# ----------------------------
# 🚨 Niveles de notificación
# ----------------------------

def send_info(symbol, message, **kwargs):
    """Mensaje informativo (sistema)."""
    text = f"ℹ️ INFO — {message}"
    print(f"{Fore.CYAN}{text}{Style.RESET_ALL}")
    formatted = format_message(symbol, message=message, evento_sistema=True, **kwargs)
    send_telegram_message(formatted)
    _guardar_log({"level": "INFO", "symbol": symbol, "msg": message, "time": datetime.now().isoformat()})


def send_warning(symbol, message, **kwargs):
    """Advertencias (trading)."""
    text = f"⚠️ WARNING — {message}"
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")
    formatted = format_message(symbol, message=message, **kwargs)
    send_telegram_message(formatted)
    _guardar_log({"level": "WARNING", "symbol": symbol, "msg": message, "time": datetime.now().isoformat()})


def send_critical(symbol, message, **kwargs):
    """Errores críticos o del sistema."""
    text = f"🚨 CRITICAL — {message}"
    print(f"{Fore.RED}{text}{Style.RESET_ALL}")
    formatted = format_message(symbol, message=message, evento_sistema=True, **kwargs)
    send_telegram_message(formatted)
    _guardar_log({"level": "CRITICAL", "symbol": symbol, "msg": message, "time": datetime.now().isoformat()})
