"""PaddleOCR-based image text extraction engine."""

from ingestion.base import OCREngine


class PaddleOCREngine(OCREngine):
    """Extracts text (Urdu/English) from images using PaddleOCR.

    The model is loaded lazily on first use.

    Install note (Urdu support):
        pip install paddlepaddle paddleocr
    The first call downloads the Urdu recognition model automatically --
    PaddleOCR ships language dictionaries and fetches the 'ur' model on
    demand. Requires network on first run.
    """

    def __init__(self, lang: str = "ur"):
        self.lang = lang
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR

            try:
                self._ocr = PaddleOCR(
                    lang=self.lang,
                    use_textline_orientation=True,
                    enable_mkldnn=False,
                    text_det_limit_side_len=640,
                )
            except (TypeError, ValueError):
                # PaddleOCR 2.x: use_textline_orientation / enable_mkldnn may not exist
                self._ocr = PaddleOCR(
                    lang=self.lang, enable_mkldnn=False, text_det_limit_side_len=640
                )
        return self._ocr

    @staticmethod
    def _parse_lines(result) -> list:
        """Return ``[(text, confidence), ...]`` from a PaddleOCR result.

        Handles both PaddleOCR 2.x (list of ``[[box, (text, conf)], ...]``)
        and 3.x (list of dict-like ``OCRResult`` with ``rec_texts`` /
        ``rec_scores``) output shapes.
        """
        lines = []
        if not result:
            return lines
        for page in result:
            if not page:
                continue
            if hasattr(page, "get"):
                texts = page.get("rec_texts") or []
                scores = page.get("rec_scores") or []
                for text, confidence in zip(texts, scores):
                    if isinstance(text, str) and text.strip():
                        lines.append((text.strip(), float(confidence)))
                continue
            for item in page:
                try:
                    text = item[1][0]
                    confidence = item[1][1]
                except (TypeError, IndexError, KeyError):
                    continue
                if isinstance(text, str) and text.strip():
                    lines.append((text.strip(), float(confidence)))
        return lines

    def extract_text(self, image_path: str) -> str:
        return "\n".join(text for text, _ in self.extract_lines(image_path))

    def extract_lines(self, image_path: str) -> list:
        ocr = self._get_ocr()
        try:
            result = ocr.ocr(image_path, cls=True)
        except (TypeError, ValueError):
            result = ocr.ocr(image_path)
        return self._parse_lines(result)
