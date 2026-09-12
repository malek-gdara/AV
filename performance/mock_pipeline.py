# src/performance/mock_pipeline.py
"""
Mock du pipeline complet (étapes 1 et 2) pour tester l'étape 3
(performance & robustesse) SANS dépendre des autres équipes.

✨ NOUVEAU : le mock simule le gain de performance selon imgsz.
    - imgsz=640 → délai complet
    - imgsz=416 → ~42% du délai
    - imgsz=320 → ~25% du délai
"""

import time
import random
from typing import Dict, Any, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

# Délais simulés à imgsz=640 (en secondes)
SIMULATED_DELAYS = {
    "detection": 0.070,     # 70 ms → YOLOv8n @ 640 sur CPU
    "tracking": 0.003,      # 3 ms → SORT simple
    "prediction": 0.005,    # 5 ms → Kalman filter
}

# ✅ NOUVEAU : facteur de gain selon imgsz
# Le temps varie en (imgsz / 640)²
IMGSZ_BASE = 640

RANDOM_SEED = 42
N_OBJECTS_MIN = 2
N_OBJECTS_MAX = 6
PROBA_EMPTY_FRAME = 0.05
PROBA_ERROR_FRAME = 0.01
CLASSES = ["person", "car", "bicycle", "motorbike", "bus", "truck"]
IMG_WIDTH = 640
IMG_HEIGHT = 480


def _imgsz_factor(imgsz: int) -> float:
    """
    Facteur de gain selon imgsz.
    - 640 → 1.0
    - 416 → 0.42
    - 320 → 0.25
    """
    return (imgsz / IMGSZ_BASE) ** 2


# ============================================================
# MOCK ÉTAPE 1 — DÉTECTION + TRACKING
# ============================================================

def mock_detection_step1(frame_id: int, timestamp: float,
                         imgsz: int = 640) -> Dict[str, Any]:
    """
    Simule la sortie de l'étape 1 (détection + tracking).
    
    ✅ NOUVEAU : accepte imgsz pour simuler le gain de performance.
    """
    # Appliquer le facteur de gain selon imgsz
    factor = _imgsz_factor(imgsz)
    delay = (SIMULATED_DELAYS["detection"] + SIMULATED_DELAYS["tracking"]) * factor
    time.sleep(delay)
    
    # Cas limite 1 : frame en erreur
    if random.random() < PROBA_ERROR_FRAME:
        return {"frame_id": frame_id, "timestamp": timestamp, "objects": []}
    
    # Cas limite 2 : frame vide
    if random.random() < PROBA_EMPTY_FRAME:
        return {"frame_id": frame_id, "timestamp": timestamp, "objects": []}
    
    # Cas normal
    n = random.randint(N_OBJECTS_MIN, N_OBJECTS_MAX)
    objects = []
    for i in range(n):
        x1 = random.uniform(0, IMG_WIDTH * 0.7)
        y1 = random.uniform(0, IMG_HEIGHT * 0.7)
        w = random.uniform(30, IMG_WIDTH * 0.3)
        h = random.uniform(30, IMG_HEIGHT * 0.3)
        x2 = min(x1 + w, IMG_WIDTH)
        y2 = min(y1 + h, IMG_HEIGHT)
        obj_id = -1 if random.random() < 0.1 else i + 1
        
        objects.append({
            "id": obj_id,
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "class": random.choice(CLASSES),
            "conf": round(random.uniform(0.5, 0.95), 2)
        })
    
    return {"frame_id": frame_id, "timestamp": timestamp, "objects": objects}


# ============================================================
# MOCK ÉTAPE 2 — PRÉDICTION
# ============================================================

def mock_prediction_step2(output_step1: Dict[str, Any]) -> Dict[str, Any]:
    """Simule la sortie de l'étape 2 (prédiction trajectoire)."""
    time.sleep(SIMULATED_DELAYS["prediction"])
    
    frame_id = output_step1["frame_id"]
    timestamp = output_step1["timestamp"] + SIMULATED_DELAYS["prediction"]
    
    predictions = []
    for obj in output_step1["objects"]:
        if obj["id"] == -1:
            continue
        
        x1, y1, x2, y2 = obj["bbox"]
        dx = random.uniform(-10, 10)
        dy = random.uniform(-10, 10)
        future_bbox = [
            round(x1 + dx, 2), round(y1 + dy, 2),
            round(x2 + dx, 2), round(y2 + dy, 2)
        ]
        risk = random.choices(
            ["none", "collision", "exit_zone", "anomaly"],
            weights=[0.7, 0.15, 0.1, 0.05]
        )[0]
        ttc = round(random.uniform(0.5, 3.0), 2) if risk == "collision" else None
        
        predictions.append({
            "id": obj["id"],
            "future_bbox": future_bbox,
            "horizon": 0.5,
            "risk": risk,
            "ttc": ttc
        })
    
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "predictions": predictions
    }


# ============================================================
# ADAPTATEUR — pour SafeInference
# ============================================================

def mock_detect_adapter(frame=None, frame_id=0, timestamp=0.0,
                         imgsz=640) -> Dict[str, Any]:
    """
    Adapte la signature du mock à celle attendue par SafeInference.
    
    ✅ NOUVEAU : transmet imgsz au mock pour simuler le gain.
    """
    return mock_detection_step1(frame_id, timestamp, imgsz=imgsz)


# ============================================================
# PIPELINE COMPLET
# ============================================================

def mock_full_pipeline(frame_id: int, timestamp: float,
                        imgsz: int = 640) -> Tuple[Dict, Dict]:
    """Simule le pipeline complet (étapes 1 + 2)."""
    out1 = mock_detection_step1(frame_id, timestamp, imgsz=imgsz)
    out2 = mock_prediction_step2(out1)
    return out1, out2


# ============================================================
# TEST RAPIDE — montre le gain selon imgsz
# ============================================================

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    
    print("🧪 Test du mock — Simulation du gain selon imgsz\n")
    print(f"{'imgsz':>6} | {'facteur':>8} | {'temps moyen':>12} | {'FPS':>6}")
    print("─" * 45)
    
    for imgsz in [640, 512, 416, 320]:
        n_frames = 20
        t_start = time.perf_counter()
        
        for i in range(n_frames):
            mock_detection_step1(i, i * 0.033, imgsz=imgsz)
        
        t_end = time.perf_counter()
        total = t_end - t_start
        per_frame_ms = (total / n_frames) * 1000
        fps = n_frames / total
        
        factor = _imgsz_factor(imgsz)
        print(f"{imgsz:>6} | {factor:>8.2f} | {per_frame_ms:>10.1f} ms | {fps:>6.1f}")
    
    print("\n✅ Le mode dégradé devrait maintenant montrer un gain de FPS.")