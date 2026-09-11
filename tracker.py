"""
tracker.py
----------
Suivi des objets (tracking + cinematique), construit sur un Detector deja
charge (pas de rechargement du modele).

Entree  : une frame (image numpy BGR).
Sortie  : liste d'objets suivis :
    {
        "track_id": int,
        "bbox": [x1, y1, x2, y2],
        "confidence": float,
        "class_id": int,
        "class_name": str,
        "velocity": (vx, vy) | None,     # px/s
        "speed": float | None,           # px/s (norme de la vitesse)
        "acceleration": (ax, ay) | None, # px/s^2
    }
C'est exactement le format attendu par le reste de l'equipe : ID,
coordonnees, vitesse -- deja fusionnes, pas besoin d'un module a part.
"""

import time
from config import CONFIG
from kinematics import TrackHistory


class Tracker:
    def __init__(self, detector, tracker_config: str = None, max_history: int = None):
        """
        Args:
            detector: instance de Detector deja creee (modele deja charge,
                      on le reutilise pour ne PAS le recharger deux fois).
        """
        self.model = detector.model
        self.conf_threshold = detector.conf_threshold
        self.target_classes = detector.target_classes
        self.tracker_config = tracker_config or CONFIG["tracker_config"]
        self.history = TrackHistory(max_history=max_history or CONFIG["max_history"])

    def update(self, frame, timestamp: float = None):
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            classes=self.target_classes,
            tracker=self.tracker_config,
            persist=True,
            verbose=False,
        )

        tracked = []
        boxes = results[0].boxes
        if boxes.id is not None:
            for box, track_id in zip(boxes, boxes.id):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                tracked.append({
                    "track_id": int(track_id),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                    "confidence": round(float(box.conf[0]), 3),
                    "class_id": int(box.cls[0]),
                    "class_name": self.model.names[int(box.cls[0])],
                })

        # fusion de la vitesse/acceleration directement dans la sortie
        tracked = self.history.update(tracked, timestamp=timestamp or time.time())
        return tracked

    def draw(self, frame, tracked_objects, show_speed: bool = True):
        import cv2
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            label = f'ID {obj["track_id"]} | {obj["class_name"]} {obj["confidence"]:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if show_speed and obj["speed"] is not None:
                cv2.putText(frame, f'{obj["speed"]:.0f} px/s', (x1, max(y1 - 25, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        return frame


# ---------- Test rapide en standalone (webcam) ----------
if __name__ == "__main__":
    import cv2
    from detector import Detector

    detector = Detector()
    tracker = Tracker(detector)
    cap = cv2.VideoCapture(CONFIG["source"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracked = tracker.update(frame)
        annotated = tracker.draw(frame, tracked)

        cv2.imshow("Tracking", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
