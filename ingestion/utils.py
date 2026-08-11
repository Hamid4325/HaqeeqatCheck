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


def strip_bidi_marks(text: str) -> str:
    """Remove bidi isolate control characters (for consumers that can't handle
    them)."""
    return text.replace(LRI, "").replace(RLI, "").replace(PDI, "")
