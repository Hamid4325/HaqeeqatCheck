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
                    use_angle_cls=True, lang=self.lang, show_log=False
                )
            except TypeError:
                self._ocr = PaddleOCR(lang=self.lang)
        return self._ocr

    @staticmethod
    def _parse_result(result) -> str:
        lines = []
        if not result:
            return ""
        for page in result:
            if not page:
                continue
            for item in page:
                try:
                    text = item[1][0]
                except (TypeError, IndexError, KeyError):
                    continue
                if isinstance(text, str) and text.strip():
                    lines.append(text.strip())
        return "\n".join(lines)

    def extract_text(self, image_path: str) -> str:
        ocr = self._get_ocr()
        try:
            result = ocr.ocr(image_path, cls=True)
        except TypeError:
            result = ocr.ocr(image_path)
        return self._parse_result(result)
