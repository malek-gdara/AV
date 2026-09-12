# src/performance/profiling.py
"""
Profiler — Mesure le temps par bloc dans le pipeline.

═══════════════════════════════════════════════════════════
🚀 GUIDE JOUR J
═══════════════════════════════════════════════════════════

Ce fichier NE NÉCESSITE AUCUN CHANGEMENT MAJEUR le jour J.
Il fonctionne aussi bien avec le mock qu'avec les vrais modules.

Le jour J, tu dois juste :
  1. Vérifier que l'import du pipeline fonctionne (voir section
     "🔴 JOUR J — IMPORTS")
  2. Ajuster les blocs profilés si tu veux mesurer plus finement

Marqueurs :
  🔵 ACTUEL          → valeur d'aujourd'hui (mode mock)
  🔴 JOUR J          → à vérifier / modifier le jour J
  ✅ ACTIVER JOUR J  → à décommenter le jour J

Usage :
    profiler = Profiler()
    with profiler.block("detection"):
        out1 = detect(frame)
    profiler.report()
═══════════════════════════════════════════════════════════
"""

import time
from contextlib import contextmanager
from typing import Dict, List

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


# ============================================================
# CLASSE PROFILER — NE PAS MODIFIER
# ============================================================

class Profiler:
    """
    Mesure le temps passé dans chaque bloc du pipeline.
    
    Usage :
        profiler = Profiler()
        with profiler.block("detection"):
            out1 = detect(frame)
        profiler.report()
    
    Chaque bloc est identifié par un nom.
    Plusieurs appels avec le même nom → durées cumulées.
    """
    
    def __init__(self, enabled=True):
        """
        Args:
            enabled : si False, le profiler ne fait rien (overhead zéro).
        """
        self.enabled = enabled
        self.durations: Dict[str, List[float]] = {}
        self.n_frames = 0
    
    @contextmanager
    def block(self, name: str):
        """Context manager pour mesurer un bloc."""
        if not self.enabled:
            yield
            return
        
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            duration_ms = (t1 - t0) * 1000
            if name not in self.durations:
                self.durations[name] = []
            self.durations[name].append(duration_ms)
    
    def frame_done(self):
        """À appeler à la fin de chaque frame."""
        if self.enabled:
            self.n_frames += 1
    
    def reset(self):
        """Réinitialise toutes les mesures."""
        self.durations = {}
        self.n_frames = 0
    
    # --------------------------------------------------------
    # STATISTIQUES
    # --------------------------------------------------------
    
    def _stats(self, values: List[float]) -> Dict[str, float]:
        """Calcule les stats pour une liste de durées."""
        if not values:
            return {"mean": 0, "p50": 0, "p95": 0, "p99": 0,
                    "min": 0, "max": 0, "total": 0, "n": 0}
        
        if _NUMPY_AVAILABLE:
            arr = np.array(values)
            return {
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "total": float(np.sum(arr)),
                "n": len(values)
            }
        else:
            sorted_v = sorted(values)
            n = len(sorted_v)
            return {
                "mean": sum(values) / n,
                "p50": sorted_v[n // 2],
                "p95": sorted_v[int(n * 0.95)],
                "p99": sorted_v[int(n * 0.99)],
                "min": min(values),
                "max": max(values),
                "total": sum(values),
                "n": n
            }
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Retourne les stats de tous les blocs."""
        return {name: self._stats(values)
                for name, values in self.durations.items()}
    
    def get_bottleneck(self) -> str:
        """Identifie le bloc qui prend le plus de temps (en moyenne)."""
        if not self.durations:
            return ""
        stats = self.get_stats()
        return max(stats.items(), key=lambda x: x[1]["mean"])[0]
    
    # --------------------------------------------------------
    # RAPPORT
    # --------------------------------------------------------
    
    def report(self, title="Profiling Report"):
        """Affiche le rapport complet."""
        print("\n" + "=" * 70)
        print(f"📊 {title} — {self.n_frames} frames")
        print("=" * 70)
        
        if not self.durations:
            print("Aucune donnée collectée.")
            return
        
        # Header
        print(f"{'Bloc':<15} | {'mean':>8} | {'p50':>8} | "
              f"{'p95':>8} | {'p99':>8} | {'n':>5}")
        print("─" * 70)
        
        stats = self.get_stats()
        total_mean = sum(s["mean"] for s in stats.values())
        
        for name, s in stats.items():
            print(f"{name:<15} | {s['mean']:>6.2f}ms | {s['p50']:>6.2f}ms | "
                  f"{s['p95']:>6.2f}ms | {s['p99']:>6.2f}ms | {s['n']:>5}")
        
        print("─" * 70)
        
        if _NUMPY_AVAILABLE:
            total_p50 = sum(s["p50"] for s in stats.values())
            total_p95 = sum(s["p95"] for s in stats.values())
            total_p99 = sum(s["p99"] for s in stats.values())
            print(f"{'TOTAL/frame':<15} | {total_mean:>6.2f}ms | "
                  f"{total_p50:>6.2f}ms | {total_p95:>6.2f}ms | "
                  f"{total_p99:>6.2f}ms |")
        
        # Goulot
        bottleneck = self.get_bottleneck()
        if bottleneck and total_mean > 0:
            pct = stats[bottleneck]["mean"] / total_mean * 100
            print(f"\n🎯 Goulot : {bottleneck} "
                  f"({stats[bottleneck]['mean']:.1f} ms, {pct:.1f}%)")
        
        # FPS estimé
        if total_mean > 0:
            fps = 1000 / total_mean
            print(f"📈 FPS estimé : {fps:.1f}")
        
        print("=" * 70)
    
    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------
    
    def to_dict(self):
        """Retourne les résultats au format dict (pour JSON)."""
        return {
            "n_frames": self.n_frames,
            "bottleneck": self.get_bottleneck(),
            "stats": self.get_stats()
        }


# ============================================================
# 🔴 JOUR J — IMPORTS DU PIPELINE
# ============================================================
# Aujourd'hui : on importe SafePipeline depuis robustness.py
# Le jour J  : même import (le pipeline réel y est déjà branché)
# ============================================================

# 🔵 ACTUEL — Import du pipeline (mock ou réel selon robustness.py)
try:
    from robustness import SafePipeline, DETECT_FN, PREDICT_FN
    _PIPELINE_AVAILABLE = True
except ImportError:
    try:
        from src.performance.robustness import (
            SafePipeline, DETECT_FN, PREDICT_FN
        )
        _PIPELINE_AVAILABLE = True
    except ImportError:
        _PIPELINE_AVAILABLE = False


# ✅ ACTIVER JOUR J — Si tu veux importer directement les vrais modules :
# from src.detection.detector import detect as real_detect
# from src.prediction.predictor import predict as real_predict


# ============================================================
# 🔴 JOUR J — CONFIGURATION DU TEST
# ============================================================
# Ajuste ces valeurs le jour J si besoin
# ============================================================

# 🔵 ACTUEL
TEST_N_FRAMES      = 50      # 🔴 JOUR J : augmenter (200-500) pour vraie mesure
TEST_TARGET_FPS    = 30      # 🔴 JOUR J : garder
TEST_ENABLE_CLAHE  = True    # 🔴 JOUR J : True si nuit, False si jour
TEST_MODE          = "granular"  # "simple" ou "granular"


# ============================================================
# TEST 1 — Profiling SIMPLE (pipeline entier)
# ============================================================

def test_simple():
    """Profile le pipeline comme un seul bloc."""
    import random
    random.seed(42)
    
    print("=" * 70)
    print("TEST SIMPLE : Profiling du pipeline entier")
    print("=" * 70)
    
    if not _PIPELINE_AVAILABLE:
        print("❌ Impossible d'importer robustness.py")
        return
    
    if DETECT_FN is None:
        print("❌ Mock indisponible")
        return
    
    pipeline = SafePipeline(
        detect_fn=DETECT_FN,
        predict_fn=PREDICT_FN,
        target_fps=TEST_TARGET_FPS,
        enable_clahe=TEST_ENABLE_CLAHE
    )
    
    profiler = Profiler(enabled=True)
    
    for i in range(TEST_N_FRAMES):
        # 🔵 ACTUEL — frame simulée
        frame = None
        if _NUMPY_AVAILABLE:
            frame = np.random.randint(10, 80, (480, 640, 3), dtype=np.uint8)
        
        with profiler.block("pipeline_total"):
            out1, out2, info = pipeline.run(
                frame, frame_id=i, timestamp=i * 0.033
            )
        
        profiler.frame_done()
    
    profiler.report("Pipeline (simple)")


# ============================================================
# TEST 2 — Profiling GRANULAIRE (chaque sous-bloc)
# ============================================================

def test_granular():
    """Profile chaque sous-bloc du pipeline."""
    import random
    random.seed(42)
    
    print("=" * 70)
    print("TEST GRANULAIRE : Profiling par sous-bloc")
    print("=" * 70)
    
    if not _PIPELINE_AVAILABLE:
        print("❌ Impossible d'importer robustness.py")
        return
    
    if DETECT_FN is None:
        print("❌ Mock indisponible")
        return
    
    pipeline = SafePipeline(
        detect_fn=DETECT_FN,
        predict_fn=PREDICT_FN,
        target_fps=TEST_TARGET_FPS,
        enable_clahe=TEST_ENABLE_CLAHE
    )
    
    profiler = Profiler(enabled=True)
    
    for i in range(TEST_N_FRAMES):
        # 🔵 ACTUEL — frame simulée
        frame = None
        if _NUMPY_AVAILABLE:
            frame = np.random.randint(10, 80, (480, 640, 3), dtype=np.uint8)
        
        # Skip si nécessaire (mode dégradé)
        if pipeline.mode.should_skip(i):
            continue
        
        # === PROFILING PAR BLOC ===
        
        # 1. CLAHE
        with profiler.block("clahe"):
            if frame is not None:
                frame_enhanced = pipeline.enhancer.enhance(frame)
            else:
                frame_enhanced = frame
        
        # 2. Détection
        with profiler.block("detection"):
            out1 = pipeline.safe_detect.infer(
                frame_enhanced,
                frame_id=i,
                timestamp=i * 0.033,
                imgsz=pipeline.mode.imgsz
            )
        
        # 3. Prédiction
        with profiler.block("prediction"):
            out2 = pipeline.safe_predict.predict(out1)
        
        # 4. Mise à jour mode (négligeable, mais mesuré)
        with profiler.block("mode_update"):
            fps = 30.0  # valeur fixe pour le test
            pipeline.mode.update(fps, frame_id=i)
        
        profiler.frame_done()
    
    profiler.report("Pipeline (granulaire)")


# ============================================================
# 🔴 JOUR J — POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    
    # 🔵 ACTUEL — Choix du mode de test
    if TEST_MODE == "simple":
        test_simple()
    elif TEST_MODE == "granular":
        test_granular()
    else:
        print(f"❌ Mode inconnu : {TEST_MODE}")
        print("   Utilise 'simple' ou 'granular'")
    
    # ✅ ACTIVER JOUR J — Pour utiliser le vrai pipeline en direct :
    # (au lieu d'importer depuis robustness.py, utilise directement
    #  tes vrais modules et frames)
    #
    # import cv2
    # from src.performance.robustness import SafePipeline
    # from src.detection.detector import detect
    # from src.prediction.predictor import predict
    #
    # pipeline = SafePipeline(detect, predict)
    # profiler = Profiler()
    # cap = cv2.VideoCapture(0)
    #
    # frame_id = 0
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     with profiler.block("pipeline"):
    #         out1, out2, info = pipeline.run(
    #             frame, frame_id, frame_id * 0.033
    #         )
    #
    #     profiler.frame_done()
    #     frame_id += 1
    #
    #     if cv2.waitKey(1) & 0xFF == 27:
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()
    # profiler.report("Vrai pipeline (live)")