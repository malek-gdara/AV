"""
config.py
---------
Charge config.json une seule fois et l'expose a tout le module.
LE JOUR J : modifiez uniquement config.json, jamais le code, pour ajuster
model_path, conf_threshold, target_classes, tracker_config, source, etc.
"""

import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config(path: str = _CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


CONFIG = load_config()
