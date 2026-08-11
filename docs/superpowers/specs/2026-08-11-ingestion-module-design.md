# Haqeeqat Check — Ingestion Module Design

Date: 2026-08-11

## Purpose

First module of **Haqeeqat Check**, an Urdu/English misinformation detector.
The ingestion module accepts an Image, Audio, or Video file and extracts all
human-readable text (speech + on-screen text) into a structured report for the
downstream analysis pipeline.

## Non-Goals

- No claim verification / fact-checking (later modules).
- No web scraping or social-media ingestion.
- No persistent storage of extracted media; temp files are cleaned up.

## Output Contract

`HaqeeqatIngestor.ingest(path) -> dict`:

```python
{
  "file_type": "video",          # "image" | "audio" | "video"
  "audio_transcript": "...",     # Whisper output ("" if image, or video with no audio)
  "ocr_text": "...",             # PaddleOCR text joined with newlines ("" if audio)
  "combined_text": "[AUDIO]: ...\n[SCREEN TEXT]: ...",
  "metadata": {
      "whisper_model": "base",
      "ocr_lang": "ur",
      "video_duration_sec": 63.2,   # audio-only files: None
      "frames_sampled": 13,          # images: 1, audio: 0
      "frames_interval_sec": 5,
      "warnings": [...]              # e.g. "no audio track found"
  }
}
```

`combined_text` is built by `bidi_safe_combine()` so Urdu (RTL) and English
(LTR) mixing does not reflow/jump in terminals or web UIs.

## Architecture

Strategy/Adapter pattern. Engines implement abstract interfaces so they can be
swapped without editing the orchestrator.

```
HaqeeqatCheck/
├── ingestion/
│   ├── __init__.py              # exports HaqeeqatIngestor, engines, errors
│   ├── base.py                  # abstract Transcriber & OCREngine ABCs + UnsupportedFormatError
│   ├── whisper_transcriber.py   # WhisperTranscriber (model_size="base")
│   ├── paddle_ocr_engine.py     # PaddleOCREngine (lang="ur")
│   ├── ingestor.py              # HaqeeqatIngestor — orchestrates everything
│   ├── utils.py                 # TempDirManager, frame sampler, dedup, bidi_safe_combine
│   └── main.py                  # CLI entry (if __name__ == "__main__")
├── tests/
│   ├── test_utils.py
│   ├── test_ingestor.py
│   └── test_engines_integration.py   # slow/auto-skip
├── requirements.txt
└── .gitignore
```

### base.py
- `class Transcriber(ABC)`: `transcribe(audio_path: str) -> str`
- `class OCREngine(ABC)`: `extract_text(image_path: str) -> str`
- `class UnsupportedFormatError(Exception)`

### whisper_transcriber.py
- `WhisperTranscriber(Transcriber)`. Lazy-loads `openai-whisper` `base` model on
  first call. Language auto-detect (handles Urdu + English). Model size is a
  constructor arg for easy upgrades (e.g. `"small"`).

### paddle_ocr_engine.py
- `PaddleOCREngine(OCREngine)`. Lazy-loads `PaddleOCR(lang="ur")` on first call.
  `extract_text(image_path)` returns text lines joined with newlines.
- Comments document Urdu setup: PaddleOCR auto-downloads the `ur` model on first
  run; requires `paddleocr` + `paddlepaddle` installed.

### ingestor.py
- `HaqeeqatIngestor(transcriber=None, ocr_engine=None)`. Defaults to
  `WhisperTranscriber()` and `PaddleOCREngine()`.
- `ingest(path)` classifies by extension and routes:
  - **image** → OCR directly
  - **audio** → transcribe directly
  - **video** → extract audio (MoviePy) + sample frames every 5 s (OpenCV) →
    transcribe + OCR → deduplicate → combine
- Builds the report dict. Handles cleanup via try/finally.

### utils.py
- `TempDirManager`: uses `tempfile.TemporaryDirectory()` / `mkdtemp` so the OS
  reclaims extracted audio/frames even on crash.
- `sample_video_frames(video_path, interval_sec=5, output_dir) -> list[path]`:
  uses `cv2.VideoCapture` (already a PaddleOCR dependency), saves one JPEG per
  interval.
- `deduplicate_text(lines) -> list[str]`: normalize (strip, collapse whitespace),
  keep first occurrence; prevents the same caption appearing 10 times.
- `bidi_safe_combine(audio_part, ocr_part) -> str`: wraps each part in Unicode
  isolates (LRI `U+2066` / RLI `U+2067`, closed by PDI `U+2069`) so RTL/LTR
  mixing doesn't reflow. Provides `strip_bidi_marks()` for consumers that can't
  handle control chars. Raw `audio_transcript` / `ocr_text` are never modified.

### main.py
- CLI: `python -m ingestion.main <file>` prints the report as JSON.
- Catches `UnsupportedFormatError` with a friendly message.

## Error Handling

- `UnsupportedFormatError` for unknown extensions or undecodable files.
- Graceful degradation within a supported format:
  - video with no audio track → `audio_transcript=""`, warning in metadata
  - video with no extractable frames → `ocr_text=""`, warning in metadata
- Clear, actionable message if Urdu PaddleOCR model or Whisper model fails to
  load/download on first use.

## Dependencies

`requirements.txt`: `openai-whisper`, `paddleocr`, `paddlepaddle`, `moviepy`,
`opencv-python`, `numpy`. Python 3.12; ffmpeg 7.1.1 already present on system.

## Testing

- **Unit (pytest):** extension classification, `deduplicate_text`,
  `bidi_safe_combine`, report building, `UnsupportedFormatError`,
  `TempDirManager` cleanup.
- **Frame sampler:** tiny OpenCV-generated video fixture, sample every 5 s,
  assert frame count and temp cleanup.
- **Integration (auto-skip if models unavailable):** Whisper on a tiny generated
  audio clip; PaddleOCR on an image with Urdu text rendered via PIL.
- **RTL:** assert `combined_text` for mixed Urdu/English contains isolate marks
  and that stripping yields clean text.

## RTL/LTR Note

Terminals and web apps can reorder mixed Urdu (RTL) + English (LTR) output. All
displayed `combined_text` is wrapped in bidi isolation controls. Consumers that
cannot process control characters should call `strip_bidi_marks()`.

## Sequence of Work

1. Scaffold package, `requirements.txt`, `.gitignore`
2. `base.py` ABCs + errors
3. `utils.py` (temp, sampler, dedup, bidi)
4. `whisper_transcriber.py`
5. `paddle_ocr_engine.py`
6. `ingestor.py`
7. `main.py`
8. Unit tests
9. Integration tests (slow/skip)
10. Manual smoke test on sample files
