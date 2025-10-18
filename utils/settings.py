# utils/settings.py

import json

def load_settings():
    with open("config/settings.json", "r") as f:
        return json.load(f)
