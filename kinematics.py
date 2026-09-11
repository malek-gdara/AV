"""
kinematics.py
-------------
Calcule vitesse et accélération par objet suivi (track_id), à partir de
l'historique de ses positions dans le temps.

Ultralytics ne fournit PAS ces valeurs directement : le tracker (ByteTrack/
BoT-SORT) utilise en interne un filtre de Kalman qui estime le mouvement,
mais cette estimation n'est pas exposée par l'API publique. On la recalcule
donc soi-même ici, à partir des positions successives des bounding boxes.

A BRANCHER dans main_integration.py :
    history = TrackHistory()
    ...
    tracked = track_objects(frame, model)
    tracked = history.update(tracked, timestamp=time.time())
    # chaque objet dans `tracked` a maintenant en plus "velocity" et "acceleration"
"""

from collections import defaultdict, deque


class TrackHistory:
    def __init__(self, max_history: int = 5):
        """
        max_history : nombre de positions passées gardées par track_id.
        5 est un bon compromis : assez pour lisser le bruit, assez peu pour
        rester réactif si l'objet change brusquement de direction.
        """
        self._history = defaultdict(lambda: deque(maxlen=max_history))

    def _center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, tracked_objects, timestamp: float):
        """
        Args:
            tracked_objects: sortie de track_objects() (liste de dicts avec track_id, bbox...)
            timestamp: temps courant en secondes (ex: time.time())

        Returns:
            La même liste, enrichie de deux champs par objet :
            - "velocity": (vx, vy) en pixels/seconde (vitesse du centre de la box)
            - "speed": norme de la vitesse, en pixels/seconde (scalaire, pratique à afficher)
            - "acceleration": (ax, ay) en pixels/seconde², ou None si pas assez d'historique
        """
        for obj in tracked_objects:
            tid = obj["track_id"]
            cx, cy = self._center(obj["bbox"])
            self._history[tid].append((timestamp, cx, cy))

            hist = self._history[tid]

            # --- Vitesse : besoin d'au moins 2 points ---
            if len(hist) >= 2:
                (t0, x0, y0), (t1, x1, y1) = hist[-2], hist[-1]
                dt = max(t1 - t0, 1e-6)  # évite division par zéro
                vx = (x1 - x0) / dt
                vy = (y1 - y0) / dt
                obj["velocity"] = (round(vx, 1), round(vy, 1))
                obj["speed"] = round((vx ** 2 + vy ** 2) ** 0.5, 1)
            else:
                obj["velocity"] = None
                obj["speed"] = None

            # --- Accélération : besoin d'au moins 3 points (2 vitesses successives) ---
            if len(hist) >= 3:
                (t0, x0, y0), (t1, x1, y1), (t2, x2, y2) = hist[-3], hist[-2], hist[-1]
                dt1 = max(t1 - t0, 1e-6)
                dt2 = max(t2 - t1, 1e-6)
                v0x, v0y = (x1 - x0) / dt1, (y1 - y0) / dt1
                v1x, v1y = (x2 - x1) / dt2, (y2 - y1) / dt2
                dt_v = max(t2 - t1, 1e-6)
                ax = (v1x - v0x) / dt_v
                ay = (v1y - v0y) / dt_v
                obj["acceleration"] = (round(ax, 1), round(ay, 1))
            else:
                obj["acceleration"] = None

        # Nettoyage optionnel : oublier les track_id qui n'apparaissent plus
        # (à activer si vous trackez sur de longues sessions pour éviter une fuite mémoire)
        active_ids = {obj["track_id"] for obj in tracked_objects}
        for tid in list(self._history.keys()):
            if tid not in active_ids:
                # on garde un peu de tolérance (l'objet peut réapparaître après occlusion)
                pass  # à durcir si besoin : del self._history[tid] après N frames d'absence

        return tracked_objects


# ---------- Exemple d'utilisation ----------
if __name__ == "__main__":
    import time
    import cv2
    from detection import load_model
    from tracking import track_objects, draw_tracks

    model = load_model()
    cap = cv2.VideoCapture(0)
    history = TrackHistory()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracked = track_objects(frame, model)
        tracked = history.update(tracked, timestamp=time.time())

        annotated = draw_tracks(frame.copy(), tracked)
        for obj in tracked:
            if obj["speed"] is not None:
                x1, y1, _, _ = obj["bbox"]
                cv2.putText(annotated, f'{obj["speed"]:.0f} px/s', (x1, max(y1 - 25, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        cv2.imshow("Tracking + Kinematics", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
