"""
detector.py
-----------
Wrapper pour YOLO (detection d'objets).

- Le modele est charge UNE SEULE FOIS a l'instanciation de Detector
  (pre-chargement, pas dans la boucle video).
- Tous les parametres par defaut viennent de config.json ; vous pouvez
  les surcharger a l'instanciation si besoin (utile pour tester rapidement
  sans toucher au fichier de config).

Entree  : une frame (image numpy BGR issue du flux video/webcam).
Sortie  : liste de detections [{bbox, confidence, class_id, class_name}, ...]
"""

from ultralytics import YOLO
from config import CONFIG


class Detector:
    def __init__(self, model_path: str = None, conf_threshold: float = None,
                 target_classes=None):
        self.model_path = model_path or CONFIG["model_path"]
        self.conf_threshold = (
            conf_threshold if conf_threshold is not None else CONFIG["conf_threshold"]
        )
        self.target_classes = (
            target_classes if target_classes is not None else CONFIG.get("target_classes")
        )
        self.model = YOLO(self.model_path)  # pre-chargement du modele

    def detect(self, frame):
        """
        Args:
            frame: image numpy (BGR, format OpenCV)

        Returns:
            list[dict]: [{"bbox":[x1,y1,x2,y2], "confidence":float,
                           "class_id":int, "class_name":str}, ...]
        """
        results = self.model(
            frame, conf=self.conf_threshold, classes=self.target_classes, verbose=False
        )
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "confidence": round(float(box.conf[0]), 3),
                "class_id": int(box.cls[0]),
                "class_name": self.model.names[int(box.cls[0])],
            })
        return detections

    def draw(self, frame, detections):
        """Dessine les bounding boxes manuellement (style custom)."""
        import cv2
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f'{det["class_name"]} {det["confidence"]:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame


# ---------- Test rapide en standalone (webcam) ----------
if __name__ == "__main__":
    import cv2

    detector = Detector()
    cap = cv2.VideoCapture(CONFIG["source"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        annotated = detector.draw(frame, detections)

        cv2.imshow("Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
