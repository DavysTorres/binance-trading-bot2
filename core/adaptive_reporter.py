# core/adaptive_reporter.py
import os
import json
from datetime import datetime
from core.notification_manager import send_info, send_warning
from modules.market_scanner_module import MarketScanner


class AdaptiveReporter:
    """
    🧠 Genera un reporte global adaptativo del ciclo operativo.
    Guarda bitácora JSON y envía resumen al Telegram si está activado.
    """

    def __init__(self, config_path="config/settings.json"):
        self.config_path = config_path
        self.load_config()
        self.logs_dir = self.config.get("logs_dir", "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

    # -----------------------
    # 🔧 Carga de configuración
    # -----------------------
    def load_config(self):
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
        self.reports_enabled = self.config.get("adaptive_reports_enabled", True)
        self.notifications_enabled = self.config.get("adaptive_notifications_enabled", True)

    # -----------------------
    # 📊 Generar reporte adaptativo
    # -----------------------
    def generar_reporte(self, stats: dict = None):
        """
        Genera y guarda un reporte adaptativo del estado actual del mercado y operaciones.
        - stats puede incluir:
          {"total_signals": int, "validated_signals": int, "open_trades": int, "closed_trades": int}
        """
        if not self.reports_enabled:
            print("📭 Reporte adaptativo desactivado en configuración.")
            return

        try:
            scanner = MarketScanner()
            sesgo = scanner.detectar_sesgo_mercado()
            intensidad = round(scanner.calcular_intensidad_mercado(), 2)
            stats = stats or {}

            reporte = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sesgo_global": sesgo,
                "intensidad": intensidad,
                "total_signals": stats.get("total_signals", 0),
                "validated_signals": stats.get("validated_signals", 0),
                "open_trades": stats.get("open_trades", 0),
                "closed_trades": stats.get("closed_trades", 0),
            }

            # Guardar JSON diario
            date_str = datetime.now().strftime("%Y%m%d")
            filename = os.path.join(self.logs_dir, f"report_{date_str}.json")
            self._guardar_json(filename, reporte)

            print(f"🧾 Reporte adaptativo guardado: {filename}")

            # Enviar al Telegram
            if self.notifications_enabled:
                self._enviar_resumen_telegram(reporte)

        except Exception as e:
            print(f"⚠️ Error generando reporte adaptativo: {e}")

    # -----------------------
    # 💾 Guardar archivo JSON
    # -----------------------
    def _guardar_json(self, filename, data):
        try:
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    prev = json.load(f)
                if isinstance(prev, list):
                    prev.append(data)
                else:
                    prev = [prev, data]
                with open(filename, "w") as f:
                    json.dump(prev, f, indent=4)
            else:
                with open(filename, "w") as f:
                    json.dump([data], f, indent=4)
        except Exception as e:
            print(f"⚠️ Error guardando reporte en {filename}: {e}")

    # -----------------------
    # 💬 Enviar resumen Telegram
    # -----------------------
    def _enviar_resumen_telegram(self, reporte):
        sesgo = reporte["sesgo_global"].upper()
        intensidad = reporte["intensidad"]
        total = reporte["total_signals"]
        validadas = reporte["validated_signals"]
        abiertas = reporte["open_trades"]
        cerradas = reporte["closed_trades"]

        resumen = (
            f"📊 *REPORTE ADAPTATIVO DEL CICLO*\n\n"
            f"🧭 Sesgo: *{sesgo}*\n"
            f"🌡️ Intensidad: *{intensidad}*\n"
            f"💡 Señales detectadas: {total}\n"
            f"✅ Señales validadas: {validadas}\n"
            f"📈 Trades abiertos: {abiertas}\n"
            f"📉 Trades cerrados: {cerradas}\n\n"
            f"🕒 {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
        )

        if intensidad >= 0.85:
            send_warning("⚠️ Alta intensidad de mercado", resumen)
        else:
            send_info("📊 Reporte adaptativo", resumen)
