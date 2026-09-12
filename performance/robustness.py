# src/performance/robustness.py
"""
Module de robustesse :
    1. AdaptiveMode      — mode dégradé automatique
    2. LowLightEnhancer  — CLAHE pour faible luminosité
    3. SafeInference     — gestion d'erreurs pour l'inférence
    4. SafePrediction    — gestion d'erreurs pour la prédiction
    5. SafePipeline      — combinaison de tout

═══════════════════════════════════════════════════════════
🚀 GUIDE JOUR J
═══════════════════════════════════════════════════════════

Le jour J, cherche "# 🔴 JOUR J" dans ce fichier et modifie
UNIQUEMENT les sections marquées. Ne touche à RIEN d'autre.

Marqueurs :
  🔵 ACTUEL          → valeur d'aujourd'hui (mode mock)
  🔴 JOUR J          → à modifier le jour J
  ✅ ACTIVER JOUR J  → à décommenter le jour J

═══════════════════════════════════════════════════════════
"""

import time

# 🔵 ACTUEL — Import cv2 nécessaire pour CLAHE
try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    print("[WARN] opencv-python non installé — CLAHE désactivé")
    print("       → pip install opencv-python")


# ============================================================
# 1. MODE DÉGRADÉ
# ============================================================

class AdaptiveMode:
    """Adapte la qualité du pipeline selon les FPS mesurés."""
    
    NORMAL = "normal"
    DEGRADED = "degraded"
    MINIMAL = "minimal"
    
    CONFIG = {
        NORMAL:   {"imgsz": 640, "skip": 1},
        DEGRADED: {"imgsz": 416, "skip": 2},
        MINIMAL:  {"imgsz": 320, "skip": 3},
    }
    
    def __init__(self, target_fps=30, hysteresis=3):
        self.target_fps = target_fps
        self.hysteresis = hysteresis
        self.mode = self.NORMAL
        self._candidate_mode = self.NORMAL
        self._candidate_count = 0
        self.fps_history = []
        self.window_size = 10
        self.mode_changes = []
        self.frames_in_mode = {self.NORMAL: 0, self.DEGRADED: 0, self.MINIMAL: 0}
    
    @property
    def imgsz(self):
        return self.CONFIG[self.mode]["imgsz"]
    
    @property
    def skip_frames(self):
        return self.CONFIG[self.mode]["skip"]
    
    def _decide_mode(self, fps):
        if fps < self.target_fps * 0.5:
            return self.MINIMAL
        elif fps < self.target_fps * 0.8:
            return self.DEGRADED
        return self.NORMAL
    
    def update(self, measured_fps, frame_id=None):
        self.fps_history.append(measured_fps)
        if len(self.fps_history) > self.window_size:
            self.fps_history.pop(0)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        self.frames_in_mode[self.mode] += 1
        target = self._decide_mode(avg_fps)
        if target == self.mode:
            self._candidate_count = 0
            return
        if target == self._candidate_mode:
            self._candidate_count += 1
        else:
            self._candidate_mode = target
            self._candidate_count = 1
        if self._candidate_count >= self.hysteresis:
            old_mode = self.mode
            self.mode = target
            self.mode_changes.append({
                "frame_id": frame_id, "from": old_mode,
                "to": target, "avg_fps": avg_fps
            })
            self._candidate_count = 0
            print(f"[MODE] {old_mode} → {target} (avg_fps={avg_fps:.1f})")
    
    def should_skip(self, frame_id):
        return (frame_id % self.skip_frames) != 0
    
    def get_stats(self):
        return {
            "mode": self.mode,
            "imgsz": self.imgsz,
            "skip_frames": self.skip_frames,
            "avg_fps": (sum(self.fps_history) / len(self.fps_history)
                        if self.fps_history else 0),
            "mode_changes": len(self.mode_changes),
            "frames_per_mode": dict(self.frames_in_mode)
        }


# ============================================================
# 2. CLAHE — Faible luminosité
# ============================================================

class LowLightEnhancer:
    """
    Améliore les frames sombres via CLAHE.
    
    - Si frame.mean() < threshold → applique CLAHE
    - Travail en LAB pour préserver les couleurs
    - Coût ~5 ms par frame quand appliqué
    """
    
    # 🔴 JOUR J — Paramètres CLAHE (à ajuster si besoin)
    DEFAULT_THRESHOLD  = 60      # Seuil de luminosité
    DEFAULT_CLIP_LIMIT = 2.0     # Force de l'amplification
    DEFAULT_TILE_SIZE  = (8, 8)  # Taille des tuiles
    
    def __init__(self, threshold=None, clip_limit=None,
                 tile_size=None, enabled=True):
        if not _CV2_AVAILABLE:
            self.enabled = False
            print("[CLAHE] Désactivé : cv2 non disponible")
            return
        
        self.threshold = threshold or self.DEFAULT_THRESHOLD
        self.clip_limit = clip_limit or self.DEFAULT_CLIP_LIMIT
        self.tile_size = tile_size or self.DEFAULT_TILE_SIZE
        self.enabled = enabled
        
        # Créer l'objet CLAHE (une seule fois)
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_size
        )
        
        # Stats
        self.n_applied = 0
        self.n_skipped = 0
        self.total_time_ms = 0.0
        
        # ✅ FIX : Warm-up CLAHE (évite le 1er appel lent ~200 ms)
        if self.enabled:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.enhance(dummy)
            self.reset_stats()
    
    def enhance(self, frame):
        """Applique CLAHE si la frame est sombre."""
        if not self.enabled or frame is None:
            return frame
        
        brightness = frame.mean()
        if brightness >= self.threshold:
            self.n_skipped += 1
            return frame
        
        t0 = time.perf_counter()
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        self.total_time_ms += (time.perf_counter() - t0) * 1000
        self.n_applied += 1
        return result
    
    def reset_stats(self):
        self.n_applied = 0
        self.n_skipped = 0
        self.total_time_ms = 0.0
    
    def get_stats(self):
        total = self.n_applied + self.n_skipped
        return {
            "enabled": self.enabled,
            "threshold": self.threshold,
            "applied": self.n_applied,
            "skipped": self.n_skipped,
            "apply_rate": self.n_applied / total if total > 0 else 0.0,
            "avg_time_ms": (self.total_time_ms / self.n_applied
                            if self.n_applied > 0 else 0.0)
        }


# ============================================================
# 3. GESTION D'ERREURS — INFÉRENCE
# ============================================================

class SafeInference:
    """Wrapper autour de l'inférence avec gestion d'erreurs."""
    
    def __init__(self, inference_fn,
                 max_consecutive_errors=10,
                 circuit_reset_delay=5.0):
        self.inference_fn = inference_fn
        self.max_consecutive_errors = max_consecutive_errors
        self.circuit_reset_delay = circuit_reset_delay
        self.n_successes = 0
        self.n_errors = 0
        self.n_consecutive_errors = 0
        self.circuit_open = False
        self.circuit_opened_at = None
    
    def _check_circuit(self):
        if not self.circuit_open:
            return
        elapsed = time.perf_counter() - self.circuit_opened_at
        if elapsed >= self.circuit_reset_delay:
            self.circuit_open = False
            self.n_consecutive_errors = 0
            print(f"[CIRCUIT] Refermé après {elapsed:.1f}s")
    
    def _build_fallback(self, frame_id, timestamp):
        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "objects": []
        }
    
    def infer(self, frame, frame_id=0, timestamp=0.0, **kwargs):
        self._check_circuit()
        if self.circuit_open:
            return self._build_fallback(frame_id, timestamp)
        try:
            result = self.inference_fn(frame, frame_id=frame_id,
                                        timestamp=timestamp, **kwargs)
            self.n_successes += 1
            self.n_consecutive_errors = 0
            return result
        except Exception as e:
            self.n_errors += 1
            self.n_consecutive_errors += 1
            print(f"[ERROR] Inférence échouée (frame {frame_id}): "
                  f"{type(e).__name__}: {e}")
            if self.n_consecutive_errors >= self.max_consecutive_errors:
                self.circuit_open = True
                self.circuit_opened_at = time.perf_counter()
                print(f"[CIRCUIT] OUVERT après "
                      f"{self.n_consecutive_errors} erreurs consécutives")
            return self._build_fallback(frame_id, timestamp)
    
    def reset(self):
        self.circuit_open = False
        self.circuit_opened_at = None
        self.n_consecutive_errors = 0
        print("[CIRCUIT] Reset manuel")
    
    def get_stats(self):
        total = self.n_successes + self.n_errors
        return {
            "successes": self.n_successes,
            "errors": self.n_errors,
            "error_rate": self.n_errors / total if total > 0 else 0.0,
            "consecutive_errors": self.n_consecutive_errors,
            "circuit_open": self.circuit_open
        }


# ============================================================
# 4. GESTION D'ERREURS — PRÉDICTION
# ============================================================

class SafePrediction:
    """Wrapper autour de la prédiction avec gestion d'erreurs."""
    
    def __init__(self, prediction_fn,
                 max_consecutive_errors=10,
                 circuit_reset_delay=5.0):
        self.prediction_fn = prediction_fn
        self.max_consecutive_errors = max_consecutive_errors
        self.circuit_reset_delay = circuit_reset_delay
        self.n_successes = 0
        self.n_errors = 0
        self.n_consecutive_errors = 0
        self.circuit_open = False
        self.circuit_opened_at = None
    
    def _check_circuit(self):
        if not self.circuit_open:
            return
        elapsed = time.perf_counter() - self.circuit_opened_at
        if elapsed >= self.circuit_reset_delay:
            self.circuit_open = False
            self.n_consecutive_errors = 0
            print(f"[CIRCUIT] Refermé après {elapsed:.1f}s")
    
    def _build_fallback(self, out1):
        return {
            "frame_id": out1.get("frame_id", 0),
            "timestamp": out1.get("timestamp", 0.0),
            "predictions": []
        }
    
    def predict(self, out1):
        self._check_circuit()
        if self.circuit_open:
            return self._build_fallback(out1)
        try:
            result = self.prediction_fn(out1)
            self.n_successes += 1
            self.n_consecutive_errors = 0
            return result
        except Exception as e:
            self.n_errors += 1
            self.n_consecutive_errors += 1
            print(f"[ERROR] Prédiction échouée "
                  f"(frame {out1.get('frame_id', '?')}): "
                  f"{type(e).__name__}: {e}")
            if self.n_consecutive_errors >= self.max_consecutive_errors:
                self.circuit_open = True
                self.circuit_opened_at = time.perf_counter()
                print(f"[CIRCUIT] OUVERT après "
                      f"{self.n_consecutive_errors} erreurs consécutives")
            return self._build_fallback(out1)
    
    def reset(self):
        self.circuit_open = False
        self.circuit_opened_at = None
        self.n_consecutive_errors = 0
        print("[CIRCUIT] Reset manuel")
    
    def get_stats(self):
        total = self.n_successes + self.n_errors
        return {
            "successes": self.n_successes,
            "errors": self.n_errors,
            "error_rate": self.n_errors / total if total > 0 else 0.0,
            "consecutive_errors": self.n_consecutive_errors,
            "circuit_open": self.circuit_open
        }


# ============================================================
# 5. SAFE PIPELINE — combine TOUT
# ============================================================

class SafePipeline:
    """
    Combine AdaptiveMode + LowLightEnhancer + SafeInference
             + SafePrediction.
    """
    
    def __init__(self, detect_fn, predict_fn,
                 target_fps=30, hysteresis=3,
                 max_consecutive_errors=10,
                 enable_clahe=True):
        self.mode = AdaptiveMode(target_fps=target_fps,
                                  hysteresis=hysteresis)
        self.enhancer = LowLightEnhancer(enabled=enable_clahe)
        self.safe_detect = SafeInference(
            detect_fn,
            max_consecutive_errors=max_consecutive_errors
        )
        self.safe_predict = SafePrediction(
            predict_fn,
            max_consecutive_errors=max_consecutive_errors
        )
        self._frame_times = []
    
    def run(self, frame, frame_id, timestamp, measure_fps=True):
        # 1. Mode dégradé : skip ?
        if self.mode.should_skip(frame_id):
            return None, None, {"skipped": True, "mode": self.mode.mode}
        
        # 2. CLAHE : améliorer si sombre
        if frame is not None:
            frame = self.enhancer.enhance(frame)
        
        # 3. Détection (safe + imgsz adaptatif)
        t0 = time.perf_counter()
        out1 = self.safe_detect.infer(
            frame, frame_id=frame_id, timestamp=timestamp,
            imgsz=self.mode.imgsz
        )
        t1 = time.perf_counter()
        
        # 4. Prédiction (safe)
        out2 = self.safe_predict.predict(out1)
        t2 = time.perf_counter()
        
        # 5. Mesure FPS
        frame_time = t2 - t0
        fps = 1.0 / frame_time if frame_time > 0 else 0.0
        self._frame_times.append(fps)
        
        if measure_fps:
            self.mode.update(fps, frame_id=frame_id)
        
        info = {
            "skipped": False,
            "fps": fps,
            "latency_ms": frame_time * 1000,
            "mode": self.mode.mode,
            "imgsz": self.mode.imgsz,
            "n_objects": len(out1.get("objects", [])),
            "n_predictions": len(out2.get("predictions", []))
        }
        return out1, out2, info
    
    def get_stats(self):
        avg_fps = (sum(self._frame_times) / len(self._frame_times)
                   if self._frame_times else 0.0)
        return {
            "mode_stats": self.mode.get_stats(),
            "clahe_stats": self.enhancer.get_stats(),
            "detect_stats": self.safe_detect.get_stats(),
            "predict_stats": self.safe_predict.get_stats(),
            "avg_fps": avg_fps
        }


# ============================================================
# 🔴 JOUR J — IMPORTS DES VRAIS MODULES
# ============================================================
# Aujourd'hui : on utilise le MOCK
# Le jour J  : on utilise les VRAIS modules
# ============================================================

# 🔵 ACTUEL — Mock
try:
    from mock_pipeline import mock_detection_step1, mock_prediction_step2
    _MOCK_AVAILABLE = True
except ImportError:
    try:
        from src.performance.mock_pipeline import (
            mock_detection_step1, mock_prediction_step2
        )
        _MOCK_AVAILABLE = True
    except ImportError:
        _MOCK_AVAILABLE = False


# ✅ FIX : Adaptateur pour aligner la signature du mock
# sur celle attendue par SafeInference
# (le mock a une signature sans `frame`, SafeInference en a besoin)
if _MOCK_AVAILABLE:
    def mock_detect_adapter(frame, frame_id=0, timestamp=0.0, imgsz=640):
        """Adapte la signature du mock à celle de SafeInference."""
        return mock_detection_step1(frame_id, timestamp)


# ✅ ACTIVER JOUR J — Décommenter ces imports le jour J :
# from src.detection.detector import detect as real_detect
# from src.prediction.predictor import predict as real_predict


# ============================================================
# 🔴 JOUR J — CHOIX DES FONCTIONS
# ============================================================

if _MOCK_AVAILABLE:
    # 🔵 ACTUEL — Mode mock (avec adaptateur)
    DETECT_FN  = mock_detect_adapter
    PREDICT_FN = mock_prediction_step2
else:
    DETECT_FN  = None
    PREDICT_FN = None

# ✅ ACTIVER JOUR J — Décommenter ces 2 lignes le jour J :
# DETECT_FN  = real_detect
# PREDICT_FN = real_predict


# ============================================================
# TEST RAPIDE
# ============================================================

if __name__ == "__main__":
    import random
    random.seed(42)
    
    # ---- TEST 1 : CLAHE ----
    print("=" * 60)
    print("TEST 1 : LowLightEnhancer (CLAHE)")
    print("=" * 60)
    
    if _CV2_AVAILABLE:
        enhancer = LowLightEnhancer(threshold=60, enabled=True)
        
        # ✅ FIX : image sombre AVEC variations (pas uniforme)
        dark = np.random.randint(10, 50, (480, 640, 3), dtype=np.uint8)
        enhanced = enhancer.enhance(dark)
        print(f"Frame sombre (mean={dark.mean():.1f})  → "
              f"mean après = {enhanced.mean():.1f}")
        
        bright = np.full((480, 640, 3), 200, dtype=np.uint8)
        enhanced = enhancer.enhance(bright)
        print(f"Frame claire (mean={bright.mean():.1f}) → "
              f"mean après = {enhanced.mean():.1f}")
        
        mid = np.full((480, 640, 3), 80, dtype=np.uint8)
        enhanced = enhancer.enhance(mid)
        print(f"Frame moyenne (mean={mid.mean():.1f}) → "
              f"mean après = {enhanced.mean():.1f}")
        
        print(f"\nStats CLAHE : {enhancer.get_stats()}")
    else:
        print("⚠️  cv2 non installé — pip install opencv-python")
    
    # ---- TEST 2 : SafeInference ----
    print("\n" + "=" * 60)
    print("TEST 2 : SafeInference — erreurs simulées")
    print("=" * 60)
    
    if DETECT_FN is None:
        print("❌ Mock indisponible")
        exit(1)
    
    call_count = {"n": 0}
    def flaky_detect(frame, frame_id=0, timestamp=0.0, imgsz=640):
        call_count["n"] += 1
        if call_count["n"] % 5 == 0:
            raise RuntimeError("Simulated failure")
        return mock_detection_step1(frame_id, timestamp)
    
    safe = SafeInference(flaky_detect, max_consecutive_errors=3)
    for i in range(15):
        out = safe.infer(None, frame_id=i, timestamp=i * 0.033)
        n = len(out["objects"])
        status = "OK" if n > 0 else "FALLBACK"
        print(f"Frame {i:2d} | {n} objets | {status}")
    print(f"\nStats : {safe.get_stats()}")
    
    # ---- TEST 3 : SafePipeline complet ----
    print("\n" + "=" * 60)
    print("TEST 3 : SafePipeline — bout en bout avec CLAHE")
    print("=" * 60)
    
    pipeline = SafePipeline(
        detect_fn=DETECT_FN,
        predict_fn=PREDICT_FN,
        target_fps=30,
        enable_clahe=True
    )
    
    for i in range(20):
        # Simuler une frame sombre AVEC variations
        if _CV2_AVAILABLE:
            frame = np.random.randint(10, 50, (480, 640, 3), dtype=np.uint8)
        else:
            frame = None
        
        out1, out2, info = pipeline.run(frame, frame_id=i,
                                         timestamp=i * 0.033)
        if info["skipped"]:
            print(f"Frame {i:2d} | SKIP")
        else:
            print(f"Frame {i:2d} | {info['n_objects']} obj | "
                  f"{info['n_predictions']} pred | "
                  f"fps={info['fps']:5.1f} | mode={info['mode']}")
    
    print("\n=== Stats finales ===")
    stats = pipeline.get_stats()
    for section, data in stats.items():
        print(f"\n[{section}]")
        if isinstance(data, dict):
            for k, v in data.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {data}")