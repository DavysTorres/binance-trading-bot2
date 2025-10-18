import json
import os

LOTE_FILE = "lote_index.json"

def cargar_lote_index():
    if os.path.exists(LOTE_FILE):
        with open(LOTE_FILE, "r") as f:
            return json.load(f).get("index", 0)
    return 0

def guardar_lote_index(index):
    with open(LOTE_FILE, "w") as f:
        json.dump({"index": index}, f)
