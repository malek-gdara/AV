"""
main_integration.py
--------------------
Point d'entree principal : boucle video temps reel = Detector + Tracker
(vitesse/acceleration deja incluses), avec un point de branchement clair
vers la partie "prediction" de l'equipe.

LE JOUR J : ajustez UNIQUEMENT config.json (modele, classes, seuil,
source video, tracker...). Ce fichier ne devrait quasiment pas bouger.
"""

import cv2
import time
from config import CONFIG
from detector import Detector
from tracker import Tracker


def send_to_prediction(tracked_objects, frame_id, timestamp):
    """
    >>> POINT D'INTEGRATION AVEC LA PARTIE PREDICTION <<<

    tracked_objects (sortie de Tracker.update()) :
    [
      {
        "track_id": 3,
        "bbox": [120, 45, 340, 210],
        "confidence": 0.87,
        "class_id": 0,
        "class_name": "person",
        "velocity": (12.5, -3.1),
        "speed": 12.9,
        "acceleration": (0.4, -0.1),
      },
      ...
    ]

    payload = {"frame_id": frame_id, "timestamp": timestamp, "objects": tracked_objects}
    # prediction_module.process(payload)   <-- a activer/adapter le jour J
    """
    pass  # a implementer avec l'equipe


def run():
    detector = Detector()      # pre-chargement du modele, une seule fois
    tracker = Tracker(detector)
    cap = cv2.VideoCapture(CONFIG["source"])

    frame_id = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        tracked = tracker.update(frame, timestamp=now)

        # --- branchement vers la prediction ---
        send_to_prediction(tracked, frame_id, now)

        if CONFIG["show_window"]:
            annotated = tracker.draw(frame.copy(), tracked, show_speed=CONFIG["show_speed"])

            fps = 1 / (now - prev_time) if now != prev_time else 0
            prev_time = now
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Vision - Detection & Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
