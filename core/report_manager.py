# core/report_manager.py
import os
import json
import csv
from datetime import datetime

# Carpeta donde se guardan los reportes
REPORTS_DIR = "logs/reportes"
os.makedirs(REPORTS_DIR, exist_ok=True)

def registrar_evento(symbol, tipo, resultado, simulada=False, mensaje=""):
    """
    Registra un evento del ciclo de reevaluación.
    - symbol: par operado
    - tipo: "reevaluacion", "ejecucion", "simulacion", "fallo"
    - resultado: "exito", "omitido", "fallo"
    - simulada: True si fue simulación
    - mensaje: texto adicional
    """
    archivo = os.path.join(REPORTS_DIR, f"eventos_{datetime.now().strftime('%Y%m%d')}.json")
    evento = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "tipo": tipo,
        "resultado": resultado,
        "simulada": simulada,
        "mensaje": mensaje
    }

    if os.path.exists(archivo):
        with open(archivo, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(evento)

    with open(archivo, "w") as f:
        json.dump(data, f, indent=4)

def generar_reporte_diario():
    """Genera un resumen con totales de eventos y rendimiento."""
    archivo = os.path.join(REPORTS_DIR, f"eventos_{datetime.now().strftime('%Y%m%d')}.json")
    if not os.path.exists(archivo):
        print("⚠️ No hay eventos registrados para generar reporte.")
        return None

    with open(archivo, "r") as f:
        eventos = json.load(f)

    total = len(eventos)
    if total == 0:
        print("⚠️ Sin eventos para reportar.")
        return None

    exitos = len([e for e in eventos if e["resultado"] == "exito"])
    fallos = len([e for e in eventos if e["resultado"] == "fallo"])
    simulaciones = len([e for e in eventos if e.get("simulada")])
    omitidos = len([e for e in eventos if e["resultado"] == "omitido"])

    tasa_exito = round((exitos / total) * 100, 2)
    tasa_fallo = round((fallos / total) * 100, 2)

    reporte = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_eventos": total,
        "exitos": exitos,
        "fallos": fallos,
        "simulaciones": simulaciones,
        "omitidos": omitidos,
        "tasa_exito_%": tasa_exito,
        "tasa_fallo_%": tasa_fallo
    }

    resumen_path = os.path.join(REPORTS_DIR, f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(resumen_path, "w") as f:
        json.dump(reporte, f, indent=4)

    # Exportar también a CSV
    generar_csv_diario(eventos)

    return reporte

def generar_csv_diario(eventos):
    """Crea o actualiza un CSV con los eventos del día."""
    csv_path = os.path.join(REPORTS_DIR, f"reporte_eventos_{datetime.now().strftime('%Y%m%d')}.csv")

    fieldnames = ["timestamp", "symbol", "tipo", "resultado", "simulada", "mensaje"]

    archivo_existe = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Escribir cabecera solo si el archivo es nuevo
        if not archivo_existe:
            writer.writeheader()

        for evento in eventos:
            writer.writerow(evento)

    print(f"📊 Reporte CSV actualizado: {csv_path}")

def enviar_resumen_console(reporte):
    """Imprime en consola el resumen de rendimiento."""
    if not reporte:
        return
    print("\n📊 === REPORTE DEL CICLO ===")
    print(f"📅 Fecha: {reporte['fecha']}")
    print(f"🔁 Total de eventos: {reporte['total_eventos']}")
    print(f"✅ Éxitos: {reporte['exitos']} | ⚠️ Fallos: {reporte['fallos']} | 💭 Simulados: {reporte['simulaciones']}")
    print(f"⏸️ Omitidos: {reporte['omitidos']}")
    print(f"📈 Tasa de Éxito: {reporte['tasa_exito_%']}% | 📉 Tasa de Fallo: {reporte['tasa_fallo_%']}%")
    print("📄 Reporte guardado en logs/reportes/\n")
