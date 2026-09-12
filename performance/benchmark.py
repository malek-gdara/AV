# src/performance/benchmark.py
"""
Benchmark YOLO — Mesure FPS, latence et percentiles.



Marqueurs :
  🔵 ACTUEL          → valeur d'aujourd'hui (test)
  🔴 JOUR J          → à modifier le jour J
  ✅ ACTIVER JOUR J  → à décommenter le jour J
═══════════════════════════════════════════════════════════
"""

import time, cv2, numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# 🔴 JOUR J — CONFIGURATION À CHANGER
# ============================================================
# Ces valeurs sont utilisées si tu lances le fichier directement.
# ============================================================

# 🔵 ACTUEL — Configuration de test aujourd'hui
DEFAULT_VIDEO   = "test_video.mp4"     # 🔴 JOUR J : mettre le chemin réel
DEFAULT_MODEL   = "yolov8n.pt"         # 🔴 JOUR J : changer si autre modèle
DEFAULT_IMGSZ   = 640                  # 🔴 JOUR J : 416 si machine lente
DEFAULT_FRAMES  = 100                  # 🔴 JOUR J : adapter selon la vidéo
DEFAULT_BATCH   = 4                    # 🔴 JOUR J : 1 pour latence minimale

# ✅ ACTIVER JOUR J — Décommenter pour flux RTSP / webcam
# DEFAULT_VIDEO = "rtsp://adresse-ip:554/stream"
# DEFAULT_VIDEO = 0                    # ← webcam (0 = webcam par défaut)


# ============================================================
# FONCTION PRINCIPALE — NE PAS MODIFIER
# ============================================================

def benchmark(video_path, model_path="yolov8n.pt", imgsz=640,
              n_frames=100, batch_size=4):
    """
    Benchmark YOLO sur une vidéo.
    
    Args:
        video_path : chemin vidéo (mp4, avi, rtsp://, ou 0 pour webcam)
        model_path : chemin du modèle YOLO
        imgsz      : résolution d'inférence
        n_frames   : nombre de frames à traiter
        batch_size : taille du batch
    
    Returns:
        dict avec les résultats (voir return à la fin)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = YOLO(model_path)
    model.to(device)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir {video_path}")
        return None
    
    # Warm-up (évite de mesurer la 1ère inférence, plus lente)
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model(dummy, imgsz=imgsz, verbose=False, device=device)
    
    times = {"capture": [], "preproc": [], "inference": [], "postproc": []}
    batch = []
    frames_processed = 0
    t_start = time.perf_counter()
    
    for i in range(n_frames):
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        t1 = time.perf_counter()
        
        frame = cv2.resize(frame, (imgsz, imgsz))
        t2 = time.perf_counter()
        
        batch.append(frame)
        times["capture"].append((t1 - t0) * 1000)
        times["preproc"].append((t2 - t1) * 1000)
        frames_processed += 1
        
        if len(batch) == batch_size:
            results = model(batch, imgsz=imgsz, verbose=False, device=device)
            t3 = time.perf_counter()
            
            # Post-traitement réel
            for r in results:
                _ = r.boxes.cpu().numpy()
            t4 = time.perf_counter()
            
            # Diviser par batch_size pour ramener au temps par frame
            times["inference"].append((t3 - t2) * 1000 / batch_size)
            times["postproc"].append((t4 - t3) * 1000 / batch_size)
            batch = []
    
    # Vider le batch restant
    if batch:
        t_batch = time.perf_counter()
        results = model(batch, imgsz=imgsz, verbose=False, device=device)
        t_after = time.perf_counter()
        times["inference"].append((t_after - t_batch) * 1000 / len(batch))
        times["postproc"].append(0)  # approximation
    
    cap.release()
    t_end = time.perf_counter()
    real_fps = frames_processed / (t_end - t_start)
    
    # ============================================================
    # AFFICHAGE
    # ============================================================
    print(f"\n=== Benchmark {model_path} @ imgsz={imgsz}, batch={batch_size} ===")
    print(f"Device: {device}")
    total = 0
    for k, v in times.items():
        if not v:
            continue
        mean = np.mean(v)
        total += mean
        print(f"{k:12s}: {mean:6.2f} ms")
    print(f"{'─' * 40}")
    print(f"{'TOTAL/frame':12s}: {total:6.2f} ms")
    print(f"{'FPS réel':12s}: {real_fps:6.1f} FPS")
    print(f"Frames: {frames_processed}/{n_frames}")
    
    # Percentiles
    print(f"\n=== Percentiles (latence) ===")
    for k, v in times.items():
        if not v:
            continue
        v = np.array(v)
        print(f"{k:12s}: p50={np.percentile(v, 50):6.2f}  "
              f"p95={np.percentile(v, 95):6.2f}  "
              f"p99={np.percentile(v, 99):6.2f}")
    
    # ============================================================
    # RETOUR DES RÉSULTATS (utilisé pour export JSON)
    # ============================================================
    result = {
        "model": model_path,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "device": device,
        "real_fps": real_fps,
        "frames_processed": frames_processed,
        "total_ms": total,
        "times": {k: float(np.mean(v)) for k, v in times.items() if v},
        "percentiles": {
            k: {
                "p50": float(np.percentile(v, 50)),
                "p95": float(np.percentile(v, 95)),
                "p99": float(np.percentile(v, 99))
            }
            for k, v in times.items() if v
        }
    }
    return result


# ============================================================
# 🔴 JOUR J — POINT D'ENTRÉE
# ============================================================
# 2 méthodes possibles :
#   1. Modifier les DEFAULT_* en haut (méthode simple) — utilisée
#   2. Passer les arguments en ligne de commande (méthode pro) — commentée
# ============================================================

if __name__ == "__main__":
    
    # ============================================================
    # 🔵 MÉTHODE SIMPLE (aujourd'hui) — valeurs codées en dur
    # ============================================================
    benchmark(
        video_path=DEFAULT_VIDEO,      # 🔴 JOUR J : voir DEFAULT_VIDEO
        model_path=DEFAULT_MODEL,      # 🔴 JOUR J : voir DEFAULT_MODEL
        imgsz=DEFAULT_IMGSZ,           # 🔴 JOUR J : voir DEFAULT_IMGSZ
        n_frames=DEFAULT_FRAMES,       # 🔴 JOUR J : voir DEFAULT_FRAMES
        batch_size=DEFAULT_BATCH       # 🔴 JOUR J : voir DEFAULT_BATCH
    )
    
    # ============================================================
    # ✅ ACTIVER JOUR J — MÉTHODE PRO (ligne de commande)
    # ============================================================
    # Décommenter ce bloc et commenter le bloc "🔵 MÉTHODE SIMPLE"
    # ci-dessus pour passer les arguments en ligne de commande.
    #
    # import argparse
    #
    # parser = argparse.ArgumentParser(description="Benchmark YOLO")
    # parser.add_argument("--video", required=True, help="Chemin vidéo")
    # parser.add_argument("--model", default="yolov8n.pt", help="Modèle YOLO")
    # parser.add_argument("--imgsz", type=int, default=640, help="Résolution")
    # parser.add_argument("--frames", type=int, default=100, help="Nb frames")
    # parser.add_argument("--batch", type=int, default=4, help="Batch size")
    # args = parser.parse_args()
    #
    # benchmark(
    #     video_path=args.video,
    #     model_path=args.model,
    #     imgsz=args.imgsz,
    #     n_frames=args.frames,
    #     batch_size=args.batch
    # )