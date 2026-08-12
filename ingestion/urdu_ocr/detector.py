"""YOLOv8 text-line detector (adapted from the UTRNet webapp app.py predict block)."""


class TextLineDetector:
    """Detects text lines in a full image with YOLOv8 (finetuned on UrduDoc)."""

    def __init__(self, weights_path, device="cpu"):
        self.weights_path = weights_path
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights_path)

    def detect(self, image):
        """Return cropped PIL images for detected lines, sorted top-to-bottom."""
        self._load()
        results = self._model.predict(
            source=image, conf=0.2, imgsz=1280, save=False, nms=True, device=self.device
        )
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        boxes.sort(key=lambda x: x[1])
        return [image.crop(box) for box in boxes]
