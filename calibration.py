# calibration.py
import time
import numpy as np
from ultralytics import YOLO

# Charger le modèle
model = YOLO("yolov8n.pt")

# Image factice (640x640)
img = np.zeros((640, 640, 3), dtype=np.uint8)

# Warm-up (5 inférences pour "chauffer" le CPU/GPU)
for _ in range(5):
    model(img, verbose=False, imgsz=640)

# Mesure sur 50 inférences
times = []
for _ in range(50):
    t0 = time.perf_counter()
    model(img, verbose=False, imgsz=640)
    times.append(time.perf_counter() - t0)

# Statistiques
mean_ms = np.mean(times) * 1000
median_ms = np.median(times) * 1000
p95_ms = np.percentile(times, 95) * 1000

print(f"=== YOLOv8n @ 640 ===")
print(f"Moyenne : {mean_ms:.1f} ms")
print(f"Médiane : {median_ms:.1f} ms")
print(f"p95     : {p95_ms:.1f} ms")
print(f"→ À mettre dans SIMULATED_DELAYS['detection'] : {mean_ms/1000:.3f}")