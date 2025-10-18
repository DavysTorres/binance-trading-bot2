# simular_oco.py

from trading.order_manager import OrderManager

def ejecutar_simulacion_orden():
    print("🧪 Iniciando simulación de orden de compra con SL/TP OCO...\n")

    manager = OrderManager()

    resultado = manager.ejecutar_orden_con_sl_tp(side="buy")

    if resultado:
        print("\n✅ Simulación completada.")
        print(f"📈 Entrada: {resultado['entrada']}")
        print(f"🛑 Stop Loss: {resultado['sl']}")
        print(f"🎯 Take Profit: {resultado['tp']}")
    else:
        print("❌ La simulación no pudo completarse.")

if __name__ == "__main__":
    ejecutar_simulacion_orden()
