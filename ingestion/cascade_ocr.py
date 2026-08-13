"""Cascade OCR engine: run the primary engine, fall back on low confidence.

A future MERGE strategy may combine both engines' results; today this stays
a pure cascade. The ``engine_used`` attribute records which engine ultimately
provided the text so the ingestor's metadata can report it.
"""

import unicodedata

from ingestion.base import OCREngine

DEFAULT_THRESHOLD = 0.5
DEFAULT_URDU_SCRIPT_FRACTION = 0.5


def _is_urdu_char(ch: str) -> bool:
    """True if the character belongs to an Arabic/Urdu script block."""
    code = ord(ch)
    return (
        0x0600 <= code <= 0x06FF
        or 0x0750 <= code <= 0x077F
        or 0xFB50 <= code <= 0xFDFF
        or 0xFE70 <= code <= 0xFEFF
    )


def urdu_script_fraction(text: str) -> float:
    """Fraction of non-whitespace characters that are Arabic/Urdu script.

    UTRNet's recognizer only emits Urdu glyphs, so genuine Urdu output scores
    near 1.0 while mangled output on English text mixes in Latin glyphs and
    scores well below that. Used to reject confidently-wrong Urdu reads.
    """
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    urdu = sum(1 for ch in chars if _is_urdu_char(ch))
    return urdu / len(chars)


class CascadeOCREngine(OCREngine):
    """Extracts text with a primary OCR engine, falling back when confident.

    The primary engine (default: UTRNet, Urdu) runs first. It is accepted only
    when it detects lines, its mean line confidence is at least
    ``confidence_threshold``, AND its output is predominantly Arabic/Urdu
    script (fraction >= ``min_urdu_script_fraction``). Otherwise the fallback
    engine (default: PaddleOCR, English) runs instead. Whichever engine
    produced the returned text is recorded in ``engine_used``.
    """

    def __init__(
        self,
        primary=None,
        fallback=None,
        confidence_threshold=DEFAULT_THRESHOLD,
        min_urdu_script_fraction=DEFAULT_URDU_SCRIPT_FRACTION,
    ):
        from ingestion.paddle_ocr_engine import PaddleOCREngine
        from ingestion.urdu_ocr import UTRNetOCREngine

        self.lang = "auto"
        self._primary = primary if primary is not None else UTRNetOCREngine()
        self._fallback = fallback if fallback is not None else PaddleOCREngine(lang="en")
        self.confidence_threshold = confidence_threshold
        self.min_urdu_script_fraction = min_urdu_script_fraction
        self.engine_used = None

    @staticmethod
    def _mean_confidence(lines) -> float:
        if not lines:
            return 0.0
        return sum(confidence for _, confidence in lines) / len(lines)

    def _run(self, engine, image_path) -> list:
        extract_lines = getattr(engine, "extract_lines", None)
        if extract_lines is not None:
            return extract_lines(image_path)
        text = engine.extract_text(image_path)
        return [(line, 1.0) for line in text.splitlines()] if text else []

    def _primary_plausible(self, lines) -> bool:
        if not lines:
            return False
        if self._mean_confidence(lines) < self.confidence_threshold:
            return False
        combined = "\n".join(text for text, _ in lines)
        return urdu_script_fraction(combined) >= self.min_urdu_script_fraction

    def extract_lines(self, image_path: str) -> list:
        """Return the winning engine's ``[(text, confidence), ...]``.

        Runs primary; falls back when the primary's output is empty, has low
        mean confidence, or is not predominantly Urdu script. Sets
        ``self.engine_used`` to the class name of the engine that produced the
        result.
        """
        try:
            primary_lines = self._run(self._primary, image_path)
        except Exception:
            primary_lines = []
        if self._primary_plausible(primary_lines):
            self.engine_used = self._primary.__class__.__name__
            return primary_lines
        fallback_lines = self._run(self._fallback, image_path)
        self.engine_used = self._fallback.__class__.__name__
        return fallback_lines

    def extract_text(self, image_path: str) -> str:
        self.engine_used = None
        return "\n".join(text for text, _ in self.extract_lines(image_path))
