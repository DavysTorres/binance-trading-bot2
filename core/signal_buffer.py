import json, os
from datetime import datetime, timedelta

class SignalBuffer:
    """
    Buffer avanzado de señales detectadas:
    - Almacena símbolo, tipo, estrategia, score_total, sesgo e intensidad.
    - Gestiona vencimiento por TTL.
    - Controla ciclo de vida: nueva → pendiente → confirmada → ejecutada/cancelada.
    """

    def __init__(self, path="signals_validated.json", ttl_minutes=30):
        self.path = path
        self.ttl = ttl_minutes
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # ============================================================
    # 🔧 Carga y guardado interno
    # ============================================================
    def _load(self):
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save(self, items):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=4, ensure_ascii=False)

    # ============================================================
    # ➕ Añadir nueva señal
    # ============================================================
    def add(self, symbol, tipo, estrategia, score_total=0.0, sesgo="neutral",
            intensidad=0.0, detalles=None):
        detalles = detalles or {}
        items = self._load()

        # eliminar duplicados del mismo símbolo y tipo
        items = [s for s in items if not (s["symbol"] == symbol and s["tipo"] == tipo)]

        nueva = {
            "symbol": symbol,
            "tipo": tipo,
            "estrategia": estrategia,
            "score_total": round(score_total, 3),
            "sesgo": sesgo,
            "intensidad": round(float(intensidad or 0), 3),
            "timestamp": datetime.now().isoformat(),
            "intentos": 0,
            "estado": "nueva",
            "detalles": detalles
        }

        items.append(nueva)
        self._save(items)
        print(f"💾 Señal añadida al buffer: {symbol} ({tipo}) | Score={score_total}")

    # ============================================================
    # 📜 Listar señales válidas (no vencidas)
    # ============================================================
    def list_fresh(self):
        now = datetime.now()
        fresh = []
        for s in self._load():
            try:
                ts = datetime.fromisoformat(s["timestamp"])
                if now - ts <= timedelta(minutes=self.ttl):
                    fresh.append(s)
            except Exception:
                continue
        return fresh

    # ============================================================
    # 🔁 Incrementar intentos
    # ============================================================
    def increment_attempts(self, symbol):
        data = self._load()
        updated = False
        for s in data:
            if s["symbol"] == symbol:
                s["intentos"] = s.get("intentos", 0) + 1
                updated = True
        if updated:
            self._save(data)
        return updated

    # ============================================================
    # 🧹 Limpieza automática
    # ============================================================
    def clean_expired(self):
        now = datetime.now()
        data = self._load()
        new_data = []
        for s in data:
            try:
                ts = datetime.fromisoformat(s["timestamp"])
                if now - ts <= timedelta(minutes=self.ttl):
                    new_data.append(s)
            except Exception:
                continue
        if len(new_data) != len(data):
            print(f"🧹 {len(data) - len(new_data)} señales expiradas eliminadas.")
        self._save(new_data)

    # ============================================================
    # 🔍 Buscar señal específica
    # ============================================================
    def find(self, symbol):
        for s in self._load():
            if s["symbol"] == symbol:
                return s
        return None

    # ============================================================
    # 🚫 Eliminar señal
    # ============================================================
    def remove(self, symbol):
        data = self._load()
        new_data = [s for s in data if s["symbol"] != symbol]
        self._save(new_data)
        print(f"🗑️ Señal eliminada del buffer: {symbol}")

    # ============================================================
    # ✏️ Actualizar campos
    # ============================================================
    def update(self, symbol, fields: dict):
        data = self._load()
        updated = False
        for s in data:
            if s["symbol"] == symbol:
                s.update(fields)
                updated = True
        if updated:
            self._save(data)
        return updated

    # ============================================================
    # ⏳ Marcar como pendiente
    # ============================================================
    def mark_pending(self, symbol, result):
        """Señal detectada pero en espera de confirmación."""
        self.update(symbol, {
            "estado": "pendiente",
            "etapa": result.get("etapa", "desconocida"),
            "score_total": result.get("score", 0),
            "timestamp": datetime.now().isoformat(),
            "intentos": 1
        })
        print(f"⏳ Señal marcada como pendiente: {symbol} (etapa={result.get('etapa')})")

    # ============================================================
    # ✅ Marcar como confirmada
    # ============================================================
    def mark_confirmed(self, symbol, result):
        """La señal ha sido confirmada y lista para ejecutar."""
        self.update(symbol, {
            "estado": "confirmada",
            "etapa": result.get("etapa", "inicio"),
            "score_total": result.get("score", 0),
            "tendencia_3m": result.get("tendencia_3m"),
            "tendencia_5m": result.get("tendencia_5m"),
            "tendencia_15m": result.get("tendencia_15m"),
            "reversa_ult2": result.get("reversa_ult2"),
            "timestamp": datetime.now().isoformat()
        })
        print(f"✅ Señal confirmada: {symbol} — Etapa={result.get('etapa')} | Score={result.get('score')}")

    # ============================================================
    # 🚀 Marcar como ejecutada
    # ============================================================
    def mark_executed(self, symbol, order_data):
        """Actualiza el buffer con información de la orden ejecutada."""
        self.update(symbol, {
            "estado": "ejecutada",
            "orden_id": order_data.get("orden_id", None),
            "precio_entrada": order_data.get("entry_price"),
            "take_profit": order_data.get("take_profit"),
            "stop_loss": order_data.get("stop_loss"),
            "resultado": order_data.get("resultado", "simulada"),
            "timestamp": datetime.now().isoformat()
        })
        print(f"🚀 Señal ejecutada: {symbol} — TP={order_data.get('take_profit')} | SL={order_data.get('stop_loss')}")

    # ============================================================
    # 🚫 Cancelar señal
    # ============================================================
    def cancel(self, symbol, reason=""):
        """Marca una señal como cancelada y guarda la razón."""
        self.update(symbol, {
            "estado": "cancelada",
            "razon": reason,
            "timestamp": datetime.now().isoformat()
        })
        print(f"🚫 Señal cancelada: {symbol} — {reason}")
