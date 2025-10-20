import sys
import io
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.signal_reevaluator import create_production_reevaluator, ReevaluationResult

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =====================
# 🔧 CONFIGURACIÓN BASE
# =====================

CONFIG = {
    "signal_reevaluator": {
        "performance": {
            "circuit_breaker_enabled": True,
            "max_errors_per_minute": 10,
            "request_timeout": 5,
            "overall_timeout": 15,
            "max_retries": 2,
            "retry_delay": 0.5
        },
        "quality_thresholds": {
            "excellent": 0.85,
            "good": 0.70,
            "poor": 0.60
        }
    },
    "reevaluation_interval_minutes": 3,
    "min_confidence_threshold": 0.7,
    "max_position_size_usdt": 1000
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/test_reevaluator.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ===============================
# 🧪 CLASE MOCK PARA DATA LOADER
# ===============================
class MockDataLoader:
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 50):
        base_price = 100
        times = pd.date_range(end=datetime.now(), periods=limit, freq="5min")

        # Tendencia alcista suave
        data = pd.DataFrame({
            "open": np.linspace(base_price, base_price * 1.02, limit),
            "high": np.linspace(base_price * 1.01, base_price * 1.025, limit),
            "low": np.linspace(base_price * 0.99, base_price * 1.015, limit),
            "close": np.linspace(base_price, base_price * 1.02, limit),
            "volume": np.random.randint(150000, 300000, size=limit)
        }, index=times)
        return data

# ==========================================
# 🚀 FUNCIÓN DE TEST PRINCIPAL ASÍNCRONA
# ==========================================
async def run_reevaluator_test():
    print("\n🔍 Iniciando test de OptimizedSignalReevaluator en entorno simulado...\n")

    reevaluator = create_production_reevaluator(CONFIG, MockDataLoader())

    # ==========================
    # 🎯 SEÑALES SIMULADAS
    # ==========================
    mock_signals = []
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

    for sym in symbols:
        signal = {
            "symbol": sym,
            "type": "LONG" if np.random.rand() > 0.5 else "SHORT",
            "timestamp": datetime.now().isoformat(),
            "composite_score": np.random.randint(65, 95),
            "adjusted_score": np.random.randint(70, 99),
            "timeframe_signals": {
                "5m": {"indicators": {"trend": "BULLISH"}},
                "15m": {"indicators": {"trend": "BULLISH"}}
            }
        }
        mock_signals.append(signal)

    # ==========================
    # 🔄 EJECUTAR RE-EVALUACIÓN
    # ==========================
    results = await reevaluator.reevaluate_signals_batch(mock_signals)

    # ==========================
    # 📊 MOSTRAR RESULTADOS
    # ==========================
    for res in results:
        print(f"\n--- Señal {res.signal['symbol']} ---")
        print(f"📈 Tipo: {res.signal['type']}")
        print(f"✅ Confirmada: {res.is_confirmed}")
        print(f"💡 Calidad Entrada: {res.entry_quality}")
        print(f"📊 Confianza Final: {res.confidence:.2f}")
        print(f"📋 Recomendación: {res.recommendation}")
        print(f"🕒 Reevaluada: {res.reevaluation_timestamp.strftime('%H:%M:%S')}")
        print(f"⚙️ Factores: {res.adjustment_factors.keys()}")

    # ==========================
    # 📈 ESTADÍSTICAS
    # ==========================
    print("\n📊 RESUMEN GLOBAL:\n", reevaluator.get_stats())
    print("\n🩺 HEALTH CHECK:\n", reevaluator.health_check())

# ==========================
# ▶️ EJECUCIÓN
# ==========================
if __name__ == "__main__":
    asyncio.run(run_reevaluator_test())
