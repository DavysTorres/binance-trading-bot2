# 🤖 Binance Trading Bot

Bot de trading automático desarrollado en **Python**, que analiza múltiples criptomonedas en **Binance**, detecta oportunidades mediante indicadores técnicos avanzados y ejecuta órdenes automáticas en el mercado **Spot** (y próximamente **Futuros**).

## 🚀 Características Principales
- Detección de tendencias y reversión con EMA, RSI, MACD, ADX, ATR y patrones.
- Ejecución automática (Market + OCO) en modo real o simulado.
- Gestión de riesgo y trailing dinámico.
- Reportes y logs detallados.
- Arquitectura modular, extensible con React y FastAPI.

## ⚙️ Instalación
```bash
pip install -r requirements.txt
```

## 🪶 Configuración Básica
```json
{
  "api_key": "TU_API_KEY",
  "api_secret": "TU_API_SECRET",
  "market_type": "SPOT",
  "sim_mode": true,
  "risk_per_trade_usdt": 5,
  "stop_loss_pct": 0.02,
  "take_profit_pct": 0.03
}
```

> Cambia `sim_mode` a `false` para operar con dinero real.

## 🧠 Ejecución
```bash
python main.py
python -m core.signal_reevaluator --once
```

## 📈 Próximos Pasos
- Integrar Binance Futures.
- Añadir interfaz React + FastAPI.
- Backtesting histórico visual.
