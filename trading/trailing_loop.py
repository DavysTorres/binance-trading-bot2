import time
import os
import sys

# Asegurar que el directorio raíz esté en el path para importar correctamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading.order_trailing_manager import revisar_trailing_stop

if __name__ == "__main__":
    print("🚀 Iniciando loop de seguimiento con trailing stop cada 60 segundos...\n")
    try:
        while True:
            revisar_trailing_stop()
            time.sleep(60)
    except KeyboardInterrupt:
        print("🛑 Loop de trailing detenido manualmente.")
