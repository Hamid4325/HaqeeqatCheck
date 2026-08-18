"""Shared helpers: temp directories, text deduplication, and RTL/LTR-safe
report formatting."""

import os
import re
import shutil
import tempfile
from typing import Iterable, List

LRI = "\u2066"  # LEFT-TO-RIGHT ISOLATE
RLI = "\u2067"  # RIGHT-TO-LEFT ISOLATE
PDI = "\u2069"  # POP DIRECTIONAL ISOLATE

_RTL_RE = re.compile(r"[\u0591-\u07FF\uFB50-\uFDFF\uFE70-\uFEFF]")


class TempDirManager:
    """Context manager that creates a temp dir and always cleans it up.

    Uses tempfile.mkdtemp() so files live under the OS temp directory; even
    if the process crashes, the OS eventually reclaims them.
    """

    def __init__(self) -> None:
        self._path: str | None = None

    def __enter__(self) -> str:
        self._path = tempfile.mkdtemp(prefix="haqeeqat_ingest_")
        return self._path

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        if self._path and os.path.isdir(self._path):
            shutil.rmtree(self._path, ignore_errors=True)
            self._path = None


def _normalize(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip()).lower()


def deduplicate_text(lines: Iterable[str]) -> List[str]:
    """Return lines in order, keeping only the first occurrence of each
    whitespace-collapsed, case-folded line."""
    seen: set = set()
    result: List[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        key = _normalize(line)
        if key not in seen:
            seen.add(key)
            result.append(line)
    return result


def _contains_rtl(text: str) -> bool:
    return bool(_RTL_RE.search(text))


def bidi_safe_combine(audio_part: str, ocr_part: str) -> str:
    """Join audio + OCR text so mixed Urdu (RTL) and English (LTR) stays
    readable. RTL segments are wrapped in RLI/PDI isolates so labels like
    '[AUDIO]:' don't jump around in terminals or web UIs."""
    sections: List[str] = []
    if audio_part.strip():
        content = audio_part
        if _contains_rtl(content):
            content = RLI + content + PDI
        sections.append("[AUDIO]: " + content)
    if ocr_part.strip():
        content = ocr_part
        if _contains_rtl(content):
            content = RLI + content + PDI
        sections.append("[SCREEN TEXT]: " + content)
    return "\n".join(sections)


def _is_urdu_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x0600 <= code <= 0x06FF
        or 0x0750 <= code <= 0x077F
        or 0xFB50 <= code <= 0xFDFF
        or 0xFE70 <= code <= 0xFEFF
    )


def is_ocr_garbled(text: str) -> bool:
    """Return True if OCR output looks like garbage (random chars, numbers only).

    Detects the common failure mode where UTRNet crashes and PaddleOCR
    English fallback produces meaningless output on an Urdu image.
    """
    stripped = text.strip()
    if not stripped:
        return True
    chars = [ch for ch in stripped if not ch.isspace()]
    if not chars:
        return True
    total = len(chars)
    urdu_count = sum(1 for ch in chars if _is_urdu_char(ch))
    alpha_count = sum(1 for ch in chars if ch.isalpha())
    digit_count = sum(1 for ch in chars if ch.isdigit())
    # If mostly digits or punctuation, it's garbage
    if digit_count > total * 0.5:
        return True
    # If very little actual alphabetic content (Urdu or Latin), it's garbage
    if alpha_count < total * 0.3:
        return True
    # If almost no Urdu and very short meaningful text, likely garbled
    if urdu_count < total * 0.05 and alpha_count < 15:
        return True
    return False


def strip_bidi_marks(text: str) -> str:
    """Remove bidi isolate control characters (for consumers that can't handle
    them)."""
    return text.replace(LRI, "").replace(RLI, "").replace(PDI, "")


def sample_video_frames(
    video_path: str,
    output_dir: str,
    interval_sec: int = 5,
) -> List[str]:
    """Extract one JPEG frame every interval_sec seconds; return saved paths.

    Uses cv2.VideoCapture (OpenCV is already a PaddleOCR dependency) so no
    extra dependency is introduced."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    saved: List[str] = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        frame_interval = max(1, int(round(fps * interval_sec)))
        index = 0
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if index % frame_interval == 0:
                out_path = os.path.join(output_dir, f"frame_{count:05d}.jpg")
                if cv2.imwrite(out_path, frame):
                    saved.append(out_path)
                count += 1
            index += 1
    finally:
        cap.release()
    return saved


def video_duration_sec(video_path: str) -> float | None:
    """Return video duration in seconds, or None if undeterminable."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps and count and fps > 0 and count > 0:
            return round(count / fps, 2)
    finally:
        cap.release()
    return None
