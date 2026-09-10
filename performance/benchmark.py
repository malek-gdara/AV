# benchmark.py — version corrigée et améliorée
import time, cv2, numpy as np
import torch
from ultralytics import YOLO

def benchmark(video_path, model_path="yolov8n.pt", imgsz=640, n_frames=100, batch_size=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = YOLO(model_path)
    model.to(device)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir {video_path}")
        return
    
    # Warm-up
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
        if not ret: break
        t1 = time.perf_counter()
        
        frame = cv2.resize(frame, (imgsz, imgsz))
        t2 = time.perf_counter()
        
        batch.append(frame)
        times["capture"].append((t1-t0)*1000)
        times["preproc"].append((t2-t1)*1000)
        frames_processed += 1
        
        if len(batch) == batch_size:
            results = model(batch, imgsz=imgsz, verbose=False, device=device)
            t3 = time.perf_counter()
            
            # Post-traitement réel
            for r in results:
                _ = r.boxes.cpu().numpy()
            t4 = time.perf_counter()
            
            # Diviser par batch_size pour ramener au temps par frame
            times["inference"].append((t3-t2)*1000 / batch_size)
            times["postproc"].append((t4-t3)*1000 / batch_size)
            batch = []
    
    # Vider le batch restant
    if batch:
        t_batch = time.perf_counter()
        results = model(batch, imgsz=imgsz, verbose=False, device=device)
        t_after = time.perf_counter()
        times["inference"].append((t_after-t_batch)*1000 / len(batch))
        times["postproc"].append(0)  # approximation
    
    cap.release()
    t_end = time.perf_counter()
    real_fps = frames_processed / (t_end - t_start)
    
    # Affichage
    print(f"\n=== Benchmark {model_path} @ imgsz={imgsz}, batch={batch_size} ===")
    print(f"Device: {device}")
    total = 0
    for k, v in times.items():
        if not v: continue
        mean = np.mean(v)
        total += mean
        print(f"{k:12s}: {mean:6.2f} ms")
    print(f"{'─'*40}")
    print(f"{'TOTAL/frame':12s}: {total:6.2f} ms")
    print(f"{'FPS réel':12s}: {real_fps:6.1f} FPS")
    print(f"Frames: {frames_processed}/{n_frames}")
    
    # Percentiles
    print(f"\n=== Percentiles (latence) ===")
    for k, v in times.items():
        if not v: continue
        v = np.array(v)
        print(f"{k:12s}: p50={np.percentile(v,50):6.2f}  "
              f"p95={np.percentile(v,95):6.2f}  "
              f"p99={np.percentile(v,99):6.2f}")

if __name__ == "__main__":
    benchmark("test_video.mp4")