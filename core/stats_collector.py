# core/stats_collector.py
import os, json, csv
from datetime import datetime

DEFAULTS = {
    "signals_buffer_path": "signals_buffer.json",         # si tu signal_buffer usa otro path, cámbialo en settings.json
    "signals_validated_path": "signals_validated.json",
    "open_orders_file": "logs/open_orders.json",
    "reportes_dir": "logs/reportes"
}

class StatsCollector:
    def __init__(self, config_path="config/settings.json"):
        self.config = self._load_json(config_path) or {}
        self.paths = {
            "buffer": self.config.get("signals_buffer_path", DEFAULTS["signals_buffer_path"]),
            "validated": self.config.get("signals_validated_path", DEFAULTS["signals_validated_path"]),
            "open_orders": self.config.get("open_orders_file", DEFAULTS["open_orders_file"]),
            "reportes_dir": self.config.get("reportes_dir", DEFAULTS["reportes_dir"])
        }

    def collect(self, overrides: dict | None = None) -> dict:
        stats = {
            "total_signals": self._count_total_signals(),
            "validated_signals": self._count_validated_signals(),
            "open_trades": self._count_open_trades(),
            "closed_trades": self._count_closed_trades_today()
        }
        if overrides:
            stats.update({k: v for k, v in overrides.items() if v is not None})
        return stats

    # ------------------ helpers ------------------
    def _load_json(self, path):
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _count_total_signals(self) -> int:
        """
        Intenta contar señales detectadas en el buffer.
        Acepta tanto lista simple como dict con clave 'signals' o similar.
        """
        data = self._load_json(self.paths["buffer"])
        if not data:
            return 0
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            # heurísticas comunes
            for key in ("signals", "senales", "buffer", "items"):
                if key in data and isinstance(data[key], list):
                    return len(data[key])
        return 0

    def _count_validated_signals(self) -> int:
        data = self._load_json(self.paths["validated"])
        if not data:
            return 0
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            # algunos guardan solo las "fuertes"
            for key in ("validas", "fuertes", "signals", "senales"):
                if key in data and isinstance(data[key], list):
                    return len(data[key])
        return 0

    def _count_open_trades(self) -> int:
        data = self._load_json(self.paths["open_orders"])
        if not data:
            return 0
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            # por si guardas {"ordenes":[...]}
            for key in ("ordenes", "orders", "open"):
                if key in data and isinstance(data[key], list):
                    return len(data[key])
        return 0

    def _count_closed_trades_today(self) -> int:
        """
        Lee el CSV diario de eventos y cuenta filas con estado 'cerrado'.
        Formato esperado: logs/reportes/reporte_eventos_YYYYMMDD.csv
        """
        date_str = datetime.now().strftime("%Y%m%d")
        ruta = os.path.join(self.paths["reportes_dir"], f"reporte_eventos_{date_str}.csv")
        if not os.path.exists(ruta):
            return 0
        count = 0
        try:
            with open(ruta, newline='', encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    estado = (row.get("estado") or row.get("status") or "").strip().lower()
                    if estado == "cerrado":
                        count += 1
        except Exception:
            return 0
        return count
