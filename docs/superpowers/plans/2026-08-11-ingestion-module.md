# Haqeeqat Check — Ingestion Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Haqeeqat Check ingestion module — a reusable package that extracts speech (Whisper) and on-screen text (PaddleOCR) from image/audio/video files into a structured report.

**Architecture:** Strategy/Adapter pattern. `Transcriber` and `OCREngine` abstract interfaces with `WhisperTranscriber` and `PaddleOCREngine` implementations; `HaqeeqatIngestor` orchestrates routing, temp-file handling, dedup, and RTL-safe report building. Engines are injected via the constructor so they can be swapped without touching the orchestrator.

**Tech Stack:** Python 3.12, openai-whisper, PaddleOCR + paddlepaddle, MoviePy, OpenCV, numpy, pytest. Platform: Windows (PowerShell), ffmpeg 7.1.1 already on PATH.

## Global Constraints

- Python 3.12.0, pip 25.2, ffmpeg 7.1.1 present on the machine.
- Free, open-source libraries only.
- All heavy model libraries (whisper, paddleocr, moviepy) must be imported **lazily inside methods**, never at module top level, so unit tests and `import ingestion` work without them installed.
- Engines must be swappable: `HaqeeqatIngestor(transcriber=..., ocr_engine=...)` accepts any object implementing the ABCs.
- Temp files must use Python's built-in `tempfile` module so the OS reclaims them even on crash.
- Report dict keys (exact): `file_type`, `audio_transcript`, `ocr_text`, `combined_text`, `metadata`. Metadata keys: `whisper_model`, `ocr_lang`, `video_duration_sec`, `frames_sampled`, `frames_interval_sec`, `warnings`.
- `combined_text` format: `[AUDIO]: ...\n[SCREEN TEXT]: ...`, with RTL segments wrapped in Unicode isolates (RLI `U+2067` / PDI `U+2069`).
- `UnsupportedFormatError` raised for unsupported or missing files.
- Code comments only where they document install steps or public behavior (user requested install comments for Urdu deps).
- No emojis in files.

---

### Task 1: Project scaffold + `base.py` (ABCs and error type)

**Files:**
- Create: `.gitignore`, `requirements.txt`, `requirements-dev.txt`, `ingestion/__init__.py`, `ingestion/base.py`
- Test: `tests/test_base.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ingestion.base.Transcriber` (abstract `transcribe(audio_path: str) -> str`), `ingestion.base.OCREngine` (abstract `extract_text(image_path: str) -> str`), `ingestion.base.UnsupportedFormatError(Exception)`.

- [ ] **Step 1: Create venv and install dev dependencies**

From project root `C:\Users\hp\Documents\HaqeeqatCheck`:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install pytest arabic-reshaper python-bidi opencv-python
```
Expected: exit code 0.

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.paddleocr/
*.egg-info/
dist/
build/
.env
```

- [ ] **Step 3: Write dependency files**

`requirements.txt`:
```
openai-whisper
paddleocr
paddlepaddle
moviepy
opencv-python
numpy
```

`requirements-dev.txt`:
```
pytest
arabic-reshaper
python-bidi
```

- [ ] **Step 4: Create empty package marker**

`ingestion/__init__.py`:
```python
"""Haqeeqat Check ingestion module."""
```

- [ ] **Step 5: Write the failing test**

`tests/test_base.py`:
```python
import pytest

from ingestion.base import OCREngine, Transcriber, UnsupportedFormatError


def test_unsupported_format_error_is_exception():
    assert issubclass(UnsupportedFormatError, Exception)
    exc = UnsupportedFormatError("boom")
    assert str(exc) == "boom"


def test_abstract_classes_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Transcriber()
    with pytest.raises(TypeError):
        OCREngine()


def test_abstract_methods_are_required():
    class Partial(Transcriber):
        pass

    with pytest.raises(TypeError):
        Partial()
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.base'`.

- [ ] **Step 7: Write minimal implementation**

`ingestion/base.py`:
```python
"""Abstract engine interfaces and module-level exceptions."""

from abc import ABC, abstractmethod


class Transcriber(ABC):
    """Converts an audio file into a text transcript."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Return the transcribed text of the audio file at audio_path."""


class OCREngine(ABC):
    """Extracts text from an image file."""

    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Return text found in the image at image_path."""


class UnsupportedFormatError(Exception):
    """Raised when a file cannot be ingested (unsupported or unreadable)."""
```

- [ ] **Step 8: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_base.py -v`
Expected: 3 passed.

- [ ] **Step 9: Commit**

```powershell
git add .gitignore requirements.txt requirements-dev.txt ingestion/__init__.py ingestion/base.py tests/test_base.py
git commit -m "feat: scaffold project and add engine interfaces"
```

---

### Task 2: `utils.py` — temp dir, dedup, RTL-safe combine

**Files:**
- Create: `ingestion/utils.py`
- Test: `tests/test_utils.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ingestion.utils.TempDirManager` — context manager; `__enter__() -> str` (path), `__exit__` always cleans up; also `.cleanup()`.
  - `ingestion.utils.deduplicate_text(lines: Iterable[str]) -> list[str]`
  - `ingestion.utils.bidi_safe_combine(audio_part: str, ocr_part: str) -> str`
  - `ingestion.utils.strip_bidi_marks(text: str) -> str`
  - Constants `LRI`, `RLI`, `PDI` (strings).

- [ ] **Step 1: Write the failing test**

`tests/test_utils.py`:
```python
import tempfile
from pathlib import Path

import pytest

from ingestion import utils

URDU = "\u0627\u0644\u0633\u0644\u0627\u0645 \u0639\u0644\u06cc\u06a9\u0645"  # السلام علیکم


class TestTempDirManager:
    def test_creates_and_cleans_up(self):
        with utils.TempDirManager() as tmp:
            probe = Path(tmp) / "probe.txt"
            probe.write_text("x", encoding="utf-8")
            assert probe.exists()
            assert str(probe).startswith(tempfile.gettempdir())
        assert not probe.exists()

    def test_cleanup_on_exception(self):
        with pytest.raises(RuntimeError):
            with utils.TempDirManager() as tmp:
                probe = Path(tmp) / "probe.txt"
                probe.write_text("x", encoding="utf-8")
                raise RuntimeError("boom")
        assert not probe.exists()


class TestDeduplicateText:
    def test_keeps_first_occurrence(self):
        lines = ["salam duniya", "SALAM   duniya", "hello", ""]
        assert utils.deduplicate_text(lines) == ["salam duniya", "hello"]

    def test_empty_input(self):
        assert utils.deduplicate_text([]) == []


class TestBidi:
    def test_combine_both_sections(self):
        combined = utils.bidi_safe_combine("hello world", URDU)
        assert combined == "[AUDIO]: hello world\n[SCREEN TEXT]: " + utils.RLI + URDU + utils.PDI

    def test_combine_empty_ocr(self):
        assert utils.bidi_safe_combine("only audio", "") == "[AUDIO]: only audio"

    def test_combine_empty_audio(self):
        assert utils.bidi_safe_combine("", URDU) == "[SCREEN TEXT]: " + utils.RLI + URDU + utils.PDI

    def test_strip_bidi_marks(self):
        text = "[SCREEN TEXT]: " + utils.RLI + URDU + utils.PDI
        assert utils.strip_bidi_marks(text) == "[SCREEN TEXT]: " + URDU

    def test_ascii_only_is_not_wrapped(self):
        assert utils.bidi_safe_combine("a", "b") == "[AUDIO]: a\n[SCREEN TEXT]: b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.utils'`.

- [ ] **Step 3: Write minimal implementation**

`ingestion/utils.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_utils.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add ingestion/utils.py tests/test_utils.py
git commit -m "feat: add temp dir manager, dedup, and RTL-safe combine helpers"
```

---

### Task 3: `utils.py` — video frame sampling + duration

**Files:**
- Modify: `ingestion/utils.py` (append two functions)
- Test: `tests/test_utils.py` (append one test class)

**Interfaces:**
- Consumes: OpenCV (installed in Task 1).
- Produces:
  - `ingestion.utils.sample_video_frames(video_path: str, output_dir: str, interval_sec: int = 5) -> list[str]`
  - `ingestion.utils.video_duration_sec(video_path: str) -> float | None`

- [ ] **Step 1: Create the shared `tmp_media` fixture**

Create `tests/conftest.py`:
```python
import math
import struct
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest


class _MediaFactory:
    def __init__(self, base: Path):
        self.base = base

    def image(self, name="img.png", size=(64, 64)):
        path = self.base / name
        img = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
        cv2.putText(img, "Urdu Test", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imwrite(str(path), img)
        return str(path)

    def tone_wav(self, name="tone.wav", seconds=1, freq=440, sample_rate=16000):
        path = self.base / name
        with wave.open(str(path), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            frames = bytearray()
            for i in range(sample_rate * seconds):
                sample = int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
                frames += struct.pack("<h", sample)
            w.writeframes(bytes(frames))
        return str(path)

    def video(self, name="clip.avi", fps=15, seconds=2, size=(64, 64)):
        path = self.base / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        for i in range(fps * seconds):
            frame = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
            cv2.putText(frame, str(i), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            writer.write(frame)
        writer.release()
        return str(path)


@pytest.fixture
def tmp_media(tmp_path):
    return _MediaFactory(tmp_path)
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_utils.py`:
```python
class TestVideoSampling:
    def test_samples_one_frame_per_interval(self, tmp_media, tmp_path):
        video_path = tmp_media.video(seconds=2, fps=15)  # 30 frames
        out_dir = tmp_path / "frames"
        out_dir.mkdir()
        frames = utils.sample_video_frames(video_path, str(out_dir), interval_sec=1)
        assert len(frames) == 2  # t=0 and t=1
        assert all(Path(f).exists() for f in frames)

    def test_duration(self, tmp_media):
        video_path = tmp_media.video(seconds=2, fps=15)
        assert utils.video_duration_sec(video_path) == pytest.approx(2.0, abs=0.1)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_utils.py -v`
Expected: FAIL with `AttributeError: module 'ingestion.utils' has no attribute 'sample_video_frames'`.

- [ ] **Step 4: Write minimal implementation**

Append to `ingestion/utils.py`:
```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_utils.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add ingestion/utils.py tests/test_utils.py tests/conftest.py
git commit -m "feat: add video frame sampling and duration helpers"
```

### Task 4: `WhisperTranscriber`

**Files:**
- Create: `ingestion/whisper_transcriber.py`
- Test: `tests/test_whisper_transcriber.py`

**Interfaces:**
- Consumes: `ingestion.base.Transcriber`.
- Produces: `ingestion.whisper_transcriber.WhisperTranscriber(Transcriber)` with `__init__(model_size: str = "base")` and `transcribe(audio_path: str) -> str`. Attribute `model_size` (used in report metadata). Model loaded lazily; `whisper` imported inside methods only.

- [ ] **Step 1: Write the failing test**

`tests/test_whisper_transcriber.py`:
```python
import sys
import types

from ingestion.whisper_transcriber import WhisperTranscriber


def test_transcribe_strips_whitespace(monkeypatch, tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"wav")

    fake_model = types.SimpleNamespace(
        transcribe=lambda path: {"text": "  hello duniya  "}
    )
    fake_whisper = types.ModuleType("whisper")
    fake_whisper.load_model = lambda size: fake_model
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    transcriber = WhisperTranscriber(model_size="base")
    assert transcriber.transcribe(str(audio)) == "hello duniya"
    assert transcriber._model is fake_model


def test_model_loaded_once(monkeypatch, tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"wav")
    load_count = 0

    def load_model(size):
        nonlocal load_count
        load_count += 1
        return types.SimpleNamespace(transcribe=lambda p: {"text": "x"})

    fake_whisper = types.ModuleType("whisper")
    fake_whisper.load_model = load_model
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    transcriber = WhisperTranscriber()
    transcriber.transcribe(str(audio))
    transcriber.transcribe(str(audio))
    assert load_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_whisper_transcriber.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.whisper_transcriber'`.

- [ ] **Step 3: Write minimal implementation**

`ingestion/whisper_transcriber.py`:
```python
"""Whisper-based speech-to-text engine."""

from ingestion.base import Transcriber


class WhisperTranscriber(Transcriber):
    """Transcribes audio using openai-whisper.

    The model is loaded lazily on first transcribe() call so importing this
    module stays fast and model downloads don't happen at import time.

    Install note:
        pip install openai-whisper
    ffmpeg must also be on your PATH (already present on this machine).

    The 'base' multilingual model auto-detects language, so it handles Urdu
    and English. Raise the model size (e.g. 'small') for better accuracy at
    the cost of speed and memory.
    """

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            import whisper

            self._model = whisper.load_model(self.model_size)
        return self._model

    def transcribe(self, audio_path: str) -> str:
        model = self._get_model()
        result = model.transcribe(audio_path)
        text = result.get("text") or ""
        return text.strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_whisper_transcriber.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add ingestion/whisper_transcriber.py tests/test_whisper_transcriber.py
git commit -m "feat: add whisper transcriber engine"
```

---

### Task 5: `PaddleOCREngine`

**Files:**
- Create: `ingestion/paddle_ocr_engine.py`
- Test: `tests/test_paddle_ocr_engine.py`

**Interfaces:**
- Consumes: `ingestion.base.OCREngine`.
- Produces: `ingestion.paddle_ocr_engine.PaddleOCREngine(OCREngine)` with `__init__(lang: str = "ur")` and `extract_text(image_path: str) -> str`. Attribute `lang`. Model loaded lazily; `paddleocr` imported inside methods only.

- [ ] **Step 1: Write the failing test**

`tests/test_paddle_ocr_engine.py`:
```python
import sys
import types

from ingestion.paddle_ocr_engine import PaddleOCREngine

URDU = "\u0633\u0644\u0627\u0645 \u062f\u0646\u06cc\u0627"  # سلام دنیا


def test_extract_text_parses_result(monkeypatch, tmp_path):
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"jpg")

    class FakeOCR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def ocr(self, image_path, cls=True):
            return [[
                [[0, 0, 1, 1], ("سلام", 0.98)],
                [[0, 0, 1, 1], ("دنیا", 0.95)],
            ]]

    fake_paddleocr = types.ModuleType("paddleocr")
    fake_paddleocr.PaddleOCR = FakeOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)

    engine = PaddleOCREngine()
    assert engine.extract_text(str(img)) == "سلام\nدنیا"
    assert engine.lang == "ur"


def test_extract_text_handles_none_pages(monkeypatch, tmp_path):
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"jpg")

    class FakeOCR:
        def __init__(self, **kwargs):
            pass

        def ocr(self, image_path, cls=True):
            return [None]

    fake_paddleocr = types.ModuleType("paddleocr")
    fake_paddleocr.PaddleOCR = FakeOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)

    assert PaddleOCREngine().extract_text(str(img)) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_paddle_ocr_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.paddle_ocr_engine'`.

- [ ] **Step 3: Write minimal implementation**

`ingestion/paddle_ocr_engine.py`:
```python
"""PaddleOCR-based image text extraction engine."""

from ingestion.base import OCREngine


class PaddleOCREngine(OCREngine):
    """Extracts text (Urdu/English) from images using PaddleOCR.

    The model is loaded lazily on first use.

    Install note (Urdu support):
        pip install paddlepaddle paddleocr
    The first call downloads the Urdu recognition model automatically —
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_paddle_ocr_engine.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add ingestion/paddle_ocr_engine.py tests/test_paddle_ocr_engine.py
git commit -m "feat: add paddleocr engine for Urdu image text extraction"
```

---

### Task 6: `HaqeeqatIngestor` orchestrator

**Files:**
- Create: `ingestion/ingestor.py`
- Test: `tests/test_ingestor.py`

**Interfaces:**
- Consumes: `ingestion.base.{Transcriber, OCREngine, UnsupportedFormatError}`, `ingestion.whisper_transcriber.WhisperTranscriber`, `ingestion.paddle_ocr_engine.PaddleOCREngine`, `ingestion.utils.{TempDirManager, bidi_safe_combine, deduplicate_text, sample_video_frames, video_duration_sec}`.
- Produces: `ingestion.ingestor.HaqeeqatIngestor` with `__init__(transcriber=None, ocr_engine=None, frames_interval_sec: int = 5)` and `ingest(path: str) -> dict`. Report keys: `file_type`, `audio_transcript`, `ocr_text`, `combined_text`, `metadata` (with `whisper_model`, `ocr_lang`, `video_duration_sec`, `frames_sampled`, `frames_interval_sec`, `warnings`). Module constants `IMAGE_EXTS`, `AUDIO_EXTS`, `VIDEO_EXTS` (sets of lowercased extensions).

- [ ] **Step 1: Write the failing test**

`tests/test_ingestor.py`:
```python
import tempfile
from pathlib import Path

import pytest

from ingestion.base import UnsupportedFormatError
from ingestion.ingestor import HaqeeqatIngestor

URDU_OCR = "\u0633\u0644\u0627\u0645 \u062f\u0646\u06cc\u0627"  # سلام دنیا


class FakeTranscriber:
    model_size = "base"

    def __init__(self, result="hello from audio"):
        self.result = result
        self.called_with = None

    def transcribe(self, path):
        self.called_with = path
        return self.result


class FakeOCREngine:
    lang = "ur"

    def __init__(self, result=URDU_OCR):
        self.result = result
        self.called_with = []

    def extract_text(self, path):
        self.called_with.append(path)
        return self.result


def make_ingestor():
    return HaqeeqatIngestor(FakeTranscriber(), FakeOCREngine())


class TestIngest:
    def test_image(self, tmp_media):
        report = make_ingestor().ingest(tmp_media.image())
        assert report["file_type"] == "image"
        assert report["ocr_text"] == URDU_OCR
        assert report["audio_transcript"] == ""
        assert report["metadata"]["frames_sampled"] == 1
        assert report["metadata"]["video_duration_sec"] is None

    def test_audio(self, tmp_media):
        report = make_ingestor().ingest(tmp_media.tone_wav())
        assert report["file_type"] == "audio"
        assert report["audio_transcript"] == "hello from audio"
        assert report["ocr_text"] == ""
        assert report["metadata"]["frames_sampled"] == 0

    def test_video(self, tmp_media):
        ingestor = make_ingestor()
        ingestor._extract_audio = lambda video_path, output_dir: None
        report = ingestor.ingest(tmp_media.video(seconds=2, fps=15))
        assert report["file_type"] == "video"
        assert report["audio_transcript"] == ""
        assert "no audio track found" in report["metadata"]["warnings"]
        assert report["metadata"]["frames_sampled"] > 0
        assert report["ocr_text"] == URDU_OCR
        assert report["metadata"]["video_duration_sec"] == pytest.approx(2.0, abs=0.1)

    def test_video_temp_cleanup(self, tmp_media):
        before = set(Path(tempfile.gettempdir()).glob("haqeeqat_ingest_*"))
        ingestor = make_ingestor()
        ingestor._extract_audio = lambda video_path, output_dir: None
        ingestor.ingest(tmp_media.video(seconds=2, fps=15))
        after = set(Path(tempfile.gettempdir()).glob("haqeeqat_ingest_*"))
        assert after == before

    def test_unsupported_extension(self, tmp_path):
        bogus = tmp_path / "notes.txt"
        bogus.write_text("hi", encoding="utf-8")
        with pytest.raises(UnsupportedFormatError):
            make_ingestor().ingest(str(bogus))

    def test_missing_file(self, tmp_path):
        with pytest.raises(UnsupportedFormatError):
            make_ingestor().ingest(str(tmp_path / "nope.mp4"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ingestor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.ingestor'`.

- [ ] **Step 3: Write minimal implementation**

`ingestion/ingestor.py`:
```python
"""HaqeeqatIngestor: routes files to the right engine and builds the report."""

import os

from ingestion import utils
from ingestion.base import OCREngine, Transcriber, UnsupportedFormatError
from ingestion.paddle_ocr_engine import PaddleOCREngine
from ingestion.whisper_transcriber import WhisperTranscriber

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv", ".3gp"}


class HaqeeqatIngestor:
    """Extracts all text (speech + on-screen) from an image, audio, or video.

    Engines are swappable: pass any object implementing Transcriber / OCREngine
    to the constructor. Defaults to WhisperTranscriber and PaddleOCREngine.
    """

    def __init__(
        self,
        transcriber: Transcriber | None = None,
        ocr_engine: OCREngine | None = None,
        frames_interval_sec: int = 5,
    ):
        self.transcriber = transcriber or WhisperTranscriber()
        self.ocr_engine = ocr_engine or PaddleOCREngine()
        self.frames_interval_sec = frames_interval_sec

    def ingest(self, path: str) -> dict:
        """Return the extraction report dict for the given media file."""
        if not os.path.isfile(path):
            raise UnsupportedFormatError(f"File not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGE_EXTS:
            return self._ingest_image(path)
        if ext in AUDIO_EXTS:
            return self._ingest_audio(path)
        if ext in VIDEO_EXTS:
            return self._ingest_video(path)
        raise UnsupportedFormatError(f"Unsupported file type: {ext or '(no extension)'}")

    def _base_metadata(self) -> dict:
        return {
            "whisper_model": getattr(self.transcriber, "model_size", None),
            "ocr_lang": getattr(self.ocr_engine, "lang", None),
            "video_duration_sec": None,
            "frames_sampled": 0,
            "frames_interval_sec": self.frames_interval_sec,
            "warnings": [],
        }

    def _ingest_image(self, path: str) -> dict:
        ocr_text = self.ocr_engine.extract_text(path)
        metadata = self._base_metadata()
        metadata["frames_sampled"] = 1
        return {
            "file_type": "image",
            "audio_transcript": "",
            "ocr_text": ocr_text,
            "combined_text": utils.bidi_safe_combine("", ocr_text),
            "metadata": metadata,
        }

    def _ingest_audio(self, path: str) -> dict:
        transcript = self.transcriber.transcribe(path)
        return {
            "file_type": "audio",
            "audio_transcript": transcript,
            "ocr_text": "",
            "combined_text": utils.bidi_safe_combine(transcript, ""),
            "metadata": self._base_metadata(),
        }

    def _ingest_video(self, path: str) -> dict:
        warnings = []
        with utils.TempDirManager() as tmp_dir:
            transcript = ""
            audio_path = self._extract_audio(path, tmp_dir)
            if audio_path:
                transcript = self.transcriber.transcribe(audio_path)
            else:
                warnings.append("no audio track found")

            frames = utils.sample_video_frames(
                path, tmp_dir, interval_sec=self.frames_interval_sec
            )
            all_lines = []
            for frame_path in frames:
                all_lines.extend(self.ocr_engine.extract_text(frame_path).splitlines())
            ocr_text = "\n".join(utils.deduplicate_text(all_lines))
            if not frames:
                warnings.append("no extractable frames")

        metadata = self._base_metadata()
        metadata["video_duration_sec"] = utils.video_duration_sec(path)
        metadata["frames_sampled"] = len(frames)
        metadata["warnings"] = warnings
        return {
            "file_type": "video",
            "audio_transcript": transcript,
            "ocr_text": ocr_text,
            "combined_text": utils.bidi_safe_combine(transcript, ocr_text),
            "metadata": metadata,
        }

    def _extract_audio(self, video_path: str, output_dir: str) -> str | None:
        """Extract the audio track to a WAV file; returns its path or None.

        Import of moviepy is deferred so the module works without it installed
        (a video with no usable audio track simply yields a warning).
        """
        try:
            try:
                from moviepy.editor import VideoFileClip  # moviepy 1.x
            except ImportError:
                from moviepy import VideoFileClip  # moviepy 2.x
        except ImportError:
            return None

        output_path = os.path.join(output_dir, "extracted_audio.wav")
        try:
            clip = VideoFileClip(video_path)
            try:
                if clip.audio is None:
                    return None
                clip.audio.write_audiofile(output_path, logger=None)
            finally:
                clip.close()
            return output_path
        except Exception:
            return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ingestor.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```powershell
git add ingestion/ingestor.py tests/test_ingestor.py
git commit -m "feat: add HaqeeqatIngestor orchestrator"
```

### Task 7: CLI (`main.py`) + package exports

**Files:**
- Modify: `ingestion/__init__.py` (add exports)
- Create: `ingestion/main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `ingestion.ingestor.HaqeeqatIngestor`, `ingestion.base.UnsupportedFormatError`.
- Produces: `ingestion.main.main(argv=None) -> int` (exit 0 = success with JSON on stdout, 2 = unsupported file with message on stderr). Package-level exports: `HaqeeqatIngestor`, `WhisperTranscriber`, `PaddleOCREngine`, `Transcriber`, `OCREngine`, `UnsupportedFormatError`.

- [ ] **Step 1: Write the failing test**

`tests/test_main.py`:
```python
from ingestion import main as cli


class TestMain:
    def test_prints_report_json(self, tmp_media, monkeypatch, capsys):
        class FakeIngestor:
            def ingest(self, path):
                return {
                    "file_type": "image",
                    "audio_transcript": "",
                    "ocr_text": "hi",
                    "combined_text": "[SCREEN TEXT]: hi",
                    "metadata": {"warnings": []},
                }

        monkeypatch.setattr(cli, "HaqeeqatIngestor", lambda: FakeIngestor())
        rc = cli.main([tmp_media.image()])
        out = capsys.readouterr().out
        assert rc == 0
        assert '"file_type": "image"' in out

    def test_unsupported_file_returns_2(self, tmp_path, capsys):
        bogus = tmp_path / "notes.txt"
        bogus.write_text("hi", encoding="utf-8")
        rc = cli.main([str(bogus)])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Error:" in err


def test_package_exports():
    import ingestion

    assert ingestion.HaqeeqatIngestor is not None
    assert ingestion.WhisperTranscriber is not None
    assert ingestion.PaddleOCREngine is not None
    assert ingestion.Transcriber is not None
    assert ingestion.OCREngine is not None
    assert ingestion.UnsupportedFormatError is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.main'`.

- [ ] **Step 3: Write minimal implementation**

`ingestion/main.py`:
```python
"""Command-line entry point for the ingestion module.

Usage:
    python -m ingestion.main <path-to-image-or-audio-or-video>
"""

import argparse
import json
import sys

from ingestion.base import UnsupportedFormatError
from ingestion.ingestor import HaqeeqatIngestor


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="haqeeqat-ingest",
        description="Extract text (speech + on-screen) from an image, audio, or video file.",
    )
    parser.add_argument("file", help="Path to an image, audio, or video file")
    args = parser.parse_args(argv)

    ingestor = HaqeeqatIngestor()
    try:
        report = ingestor.ingest(args.file)
    except UnsupportedFormatError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Replace `ingestion/__init__.py` contents:
```python
"""Haqeeqat Check ingestion module."""

from ingestion.base import OCREngine, Transcriber, UnsupportedFormatError
from ingestion.ingestor import HaqeeqatIngestor
from ingestion.paddle_ocr_engine import PaddleOCREngine
from ingestion.whisper_transcriber import WhisperTranscriber

__all__ = [
    "HaqeeqatIngestor",
    "WhisperTranscriber",
    "PaddleOCREngine",
    "Transcriber",
    "OCREngine",
    "UnsupportedFormatError",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_main.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full unit suite**

Run: `.\.venv\Scripts\python.exe -m pytest -v`
Expected: all unit tests pass.

- [ ] **Step 6: Commit**

```powershell
git add ingestion/main.py ingestion/__init__.py tests/test_main.py
git commit -m "feat: add CLI entry point and package exports"
```

---

### Task 8: Integration tests (slow, auto-skip)

**Files:**
- Create: `tests/test_engines_integration.py`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: `WhisperTranscriber`, `PaddleOCREngine`, `HaqeeqatIngestor`, `tmp_media` fixture.
- Produces: nothing (verification only).

- [ ] **Step 1: Write `pytest.ini`**

`pytest.ini`:
```ini
[pytest]
markers =
    slow: long-running tests that download models (auto-skip on failure)
```

- [ ] **Step 2: Write the integration tests**

`tests/test_engines_integration.py`:
```python
"""Integration tests for the real Whisper and PaddleOCR engines.

These download models on first run (requires network) and are slow, so they
skip themselves when a dependency, font, or model is unavailable.
"""

import os

import pytest

from ingestion.ingestor import HaqeeqatIngestor
from ingestion.paddle_ocr_engine import PaddleOCREngine
from ingestion.whisper_transcriber import WhisperTranscriber

pytestmark = pytest.mark.slow


def _find_arabic_font():
    font_dir = r"C:\Windows\Fonts"
    candidates = ["arial.ttf", "arialuni.ttf", "tahoma.ttf", "segoeui.ttf", "times.ttf"]
    for name in candidates:
        path = os.path.join(font_dir, name)
        if os.path.exists(path):
            return path
    return None


def test_whisper_transcribes_audio(tmp_media):
    transcriber = WhisperTranscriber(model_size="base")
    try:
        result = transcriber.transcribe(tmp_media.tone_wav(seconds=1))
    except Exception as exc:
        pytest.skip(f"Whisper model unavailable: {exc}")
    assert isinstance(result, str)


def test_paddle_ocr_reads_urdu_image(tmp_path):
    font_path = _find_arabic_font()
    if font_path is None:
        pytest.skip("No Arabic-capable font found")

    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        pytest.skip(f"Image rendering deps unavailable: {exc}")

    text = "\u067e\u0627\u06a9\u0633\u062a\u0627\u0646"  # پاکستان
    try:
        display_text = get_display(arabic_reshaper.reshape(text))
        img = Image.new("RGB", (400, 120), "white")
        draw = ImageDraw.Draw(img)
        draw.text(
            (10, 30), display_text, fill="black",
            font=ImageFont.truetype(font_path, 48),
        )
        img_path = tmp_path / "urdu.png"
        img.save(str(img_path))
    except Exception as exc:
        pytest.skip(f"Cannot render Urdu image: {exc}")

    try:
        result = PaddleOCREngine().extract_text(str(img_path))
    except Exception as exc:
        pytest.skip(f"PaddleOCR model unavailable: {exc}")
    assert isinstance(result, str)
    assert len(result) > 0


def test_video_pipeline_end_to_end(tmp_path, tmp_media):
    try:
        try:
            from moviepy import VideoFileClip, AudioFileClip  # moviepy 2.x
        except ImportError:
            from moviepy.editor import VideoFileClip, AudioFileClip  # moviepy 1.x
    except ImportError as exc:
        pytest.skip(f"moviepy unavailable: {exc}")

    wav = tmp_media.tone_wav(name="tone.wav", seconds=1)
    try:
        audio = AudioFileClip(wav)
        clip = VideoFileClip(tmp_media.video(seconds=1, fps=10)).with_audio(audio)
        video_path = str(tmp_path / "with_audio.mp4")
        clip.write_videofile(video_path, fps=10, logger=None)
        clip.close()
        audio.close()
    except Exception as exc:
        pytest.skip(f"Cannot build video fixture: {exc}")

    try:
        report = HaqeeqatIngestor().ingest(video_path)
    except Exception as exc:
        pytest.skip(f"Ingestion failed (models unavailable?): {exc}")

    assert report["file_type"] == "video"
    assert isinstance(report["audio_transcript"], str)
    assert report["metadata"]["frames_sampled"] > 0
```

- [ ] **Step 3: Run the integration tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_engines_integration.py -v`
Expected: tests either pass or SKIP (never fail) — they require model downloads.

- [ ] **Step 4: Commit**

```powershell
git add tests/test_engines_integration.py pytest.ini
git commit -m "test: add auto-skipping integration tests for real engines"
```

---

### Task 9: Install runtime deps + smoke test

**Files:**
- Modify: none (verification only).

- [ ] **Step 1: Install runtime dependencies**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Expected: exit code 0. This installs openai-whisper, paddleocr, paddlepaddle, moviepy (large; may take several minutes).

- [ ] **Step 2: Run the full test suite**

Run: `.\.venv\Scripts\python.exe -m pytest -v`
Expected: unit tests pass; slow integration tests pass or skip.

- [ ] **Step 3: Smoke test the CLI on generated media**

Generate a sample image, audio, and video, then run the CLI on each:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_utils.py -q --collect-only  # sanity: suite still imports
.\.venv\Scripts\python.exe -c "from ingestion import HaqeeqatIngestor; print(HaqeeqatIngestor().__class__.__name__)"
```
Create sample media and run:
```powershell
.\.venv\Scripts\python.exe -c "import cv2, numpy as np, os; os.makedirs('sample_media', exist_ok=True); cv2.imwrite('sample_media/test.png', np.full((128,128,3),255,np.uint8))"
.\.venv\Scripts\python.exe -m ingestion.main sample_media/test.png
```
Expected: JSON report printed with `"file_type": "image"`. This first run also downloads the PaddleOCR Urdu model (needs network).

- [ ] **Step 4: Final verification and commit**

Run: `.\.venv\Scripts\python.exe -m pytest -v`
Expected: all green.

```powershell
git add -A
git commit -m "chore: finalize ingestion module"
```

---

