"""End-to-end Urdu OCR engine: YOLOv8 detection + UTRNet recognition."""

import os

from ingestion.base import OCREngine

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models"
)
RECOGNIZER_WEIGHTS = "best_norm_ED.pth"
DETECTOR_WEIGHTS = "yolov8m_UrduDoc.pt"


class UTRNetOCREngine(OCREngine):
    """Extracts Urdu text from an image using YOLOv8 detection + UTRNet recognition.

    Models are loaded lazily on first use and read from ``model_dir``
    (default: ``<repo>/models/``). Download them with
    ``python tools/download_urdu_models.py``.
    """

    def __init__(self, lang="ur", model_dir=None, device="auto", detector=None, recognizer=None):
        self.lang = lang
        self.model_dir = model_dir or MODELS_DIR
        self.device = device
        self._detector = detector
        self._recognizer = recognizer

    def _resolve_device(self):
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_models(self):
        if self._recognizer is not None:
            return
        rec_path = os.path.join(self.model_dir, RECOGNIZER_WEIGHTS)
        det_path = os.path.join(self.model_dir, DETECTOR_WEIGHTS)
        missing = [p for p in (rec_path, det_path) if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "UTRNet OCR model files missing: %s. Run "
                "`python tools/download_urdu_models.py` to fetch them." % ", ".join(missing)
            )
        device = self._resolve_device()
        from ingestion.urdu_ocr.detector import TextLineDetector
        from ingestion.urdu_ocr.recognizer import UTRNetRecognizer

        self._recognizer = UTRNetRecognizer(rec_path, device=device)
        self._detector = TextLineDetector(det_path, device=device)

    def extract_text(self, image_path: str) -> str:
        return "\n".join(text for text, _ in self.extract_lines(image_path))

    def extract_lines(self, image_path: str):
        """Return ``[(text, confidence), ...]`` for each detected text line."""
        self._load_models()
        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
        except OSError:
            return []
        crops = self._detector.detect(image)
        lines = []
        for crop in crops:
            text, confidence = self._recognizer.recognize_with_confidence(crop)
            if text.strip():
                lines.append((text.strip(), confidence))
        return lines
