# Haqeeqat Check — UTRNet + YOLOv8 Urdu OCR Engine Design

Date: 2026-08-12

## Purpose

Replace the failing OCR path in Haqeeqat Check's ingestion module with the
official UTRNet end-to-end Urdu OCR pipeline (YOLOv8 text-line detection
finetuned on UrduDoc + UTRNet recognition), validated empirically against the
user's test image `images.jpg` (expected text: شکریہ پاکستان).

## Background / Why

- `PaddleOCREngine` fails on Nastaliq Urdu (returned `مکوان`/`التان` garbage for
  an image saying شکریہ پاکستان) and crashed natively (exit 0xC0000005) on
  `images.jpg`.
- `EasyOCR` was validated next via `tools/validate_ocr.py` and FAILED: returned
  `(\nتان` for `images.jpg`. The plan's gate directive was to pivot to UTRNet.
- UTRNet (ICDAR 2023, CC BY-NC-SA 4.0) is a text-LINE recognizer — a full image
  needs a separate detection stage. The developers' official
  `End-To-End-Urdu-OCR-WebApp` pairs YOLOv8 (finetuned on UrduDoc) for detection
  with UTRNet for recognition. We follow that official pipeline.

## Decisions (user-directed, 2026-08-12)

- **Pipeline:** YOLOv8 (`yolov8m_UrduDoc.pt`) line detection + UTRNet
  (`best_norm_ED.pth`) recognition, exactly as the developers' webapp does.
- **Integration:** vendor the minimal inference code into
  `ingestion/urdu_ocr/` (adapted for torch 2.13 / Python 3.12).
- **Structure:** composable units — `detector.py`, `recognizer.py`,
  `converter.py`, `model.py`, `unet.py` composed by a thin
  `UTRNetOCREngine`.
- **Default engine:** UTRNet becomes the default OCR engine in
  `HaqeeqatIngestor` ONLY after the validation gate PASSES. PaddleOCR stays as
  a constructor-injectable fallback.
- **License:** CC BY-NC-SA 4.0 (non-commercial) accepted for the project;
  attribution noted in a vendored `NOTICE.md`.
- **easyocr:** dropped from `requirements.txt` (validation-only, failed).
- **Scope:** this milestone covers recognition-quality validation + the
  UTRNet engine + default swap. No other OCR engine changes.

## Non-Goals

- No training/fine-tuning of UTRNet or YOLOv8.
- No CUDA-specific work (CPU inference is the target).
- No ContourNet detector (official webapp uses YOLOv8).
- No changes to the `OCREngine.extract_text(image_path) -> str` contract or the
  ingestor's report schema.

## Package Layout & Vendored Code

New package `ingestion/urdu_ocr/`, adapted from the official webapp
(https://github.com/abdur75648/End-To-End-Urdu-OCR-WebApp):

```
ingestion/urdu_ocr/
├── __init__.py              # exports UTRNetOCREngine
├── NOTICE.md                # upstream attribution + CC BY-NC-SA 4.0 terms
├── UrduGlyphs.txt           # 540-char vocab, verbatim from webapp
├── converter.py             # CTCLabelConverter + NormalizePAD   (from utils.py)
├── unet.py                  # UNet backbone                      (from modules/cnn/unet.py)
├── model.py                 # UNet + dropout + 2xBiLSTM + Linear (from model.py, modules/sequence_modeling.py, modules/dropout_layer.py)
├── recognizer.py            # UTRNetRecognizer                    (from read.py)
├── detector.py              # TextLineDetector                    (from app.py predict block)
└── urdu_ocr_engine.py       # UTRNetOCREngine(OCREngine): composes detector + recognizer
```

Dropped from upstream: `app.py` (Gradio UI), `modules/prediction.py` (unused —
the webapp `Model` uses `nn.Linear` directly), example images, and bounding-box
drawing. Every vendored file gets a short header noting its upstream origin.

## Engine Behavior & Contract

### UTRNetOCREngine(OCREngine) — `urdu_ocr_engine.py`

- `__init__(self, lang="ur", model_dir=None, device="auto")`
  - `lang` -> `"ur"` (read by `_base_metadata` in `ingestor.py`)
  - `model_dir` -> default `<repo>/models/` (gitignored)
  - `device` -> `"auto"` (cuda if available else cpu); overridable for tests
- `extract_text(image_path) -> str`:
  1. Lazy-load detector + recognizer on first call only (heavy torch imports
     stay inside methods, matching the existing lazy-import convention).
  2. Open image via PIL (RGB).
  3. `detector.detect(image)` -> YOLO `predict(conf=0.2, imgsz=1280, nms=True)`,
     boxes sorted by y, cropped as PIL images.
  4. For each crop: `recognizer.recognize(crop)` -> text string.
  5. Join with `"\n"`. Return `""` if no lines detected.
- Missing-model error: if `best_norm_ED.pth` / `yolov8m_UrduDoc.pt` are absent
  from `model_dir`, raise a clear error naming the missing file and pointing to
  `tools/download_urdu_models.py`.
- Empty/unreadable image: return `""` (graceful, like `PaddleOCREngine`).

### UTRNetRecognizer — `recognizer.py`

Owns model load, preprocessing (grayscale -> `FLIP_LEFT_RIGHT` mirror -> resize
height 32, width <=400 -> `NormalizePAD` to 1x32x400), forward pass, CTC decode.

### TextLineDetector — `detector.py`

Owns YOLO load + predict + crop.

## Model Acquisition

New script `tools/download_urdu_models.py` (stdlib `urllib` only, Windows-
friendly):

- Downloads into `<repo>/models/`:
  - `best_norm_ED.pth` (41 MB, recognition)
  - `yolov8m_UrduDoc.pt` (49.7 MB, detection)
  - Base URL: `https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet/resolve/main/<file>`
- Skips existing valid-size files; `--force` re-downloads; prints progress.
- `models/` added to `.gitignore` (never commit ~91 MB of weights).

## Dependencies

- Add `ultralytics` to `requirements.txt` (YOLOv8 runtime). torch 2.13.0+cpu,
  torchvision, Pillow, numpy already installed — verify pip keeps them and does
  not force a torch reinstall.
- Remove `easyocr` from `requirements.txt`. The harness already handles a
  missing easyocr gracefully (records `"unavailable"`).
- Verify ultralytics' numpy/opencv requirements against installed versions at
  install time.

## Validation Harness

Extend `tools/validate_ocr.py`:

- After the PaddleOCR baseline and EasyOCR, run UTRNet via lazy-imported
  `UTRNetOCREngine.extract_text`.
- If `models/` is missing, record a clear error pointing to the download script.
- Run on `images.jpg` (expected: شکریہ پاکستان); append to
  `.superpowers/sdd/ocr-validation.md` with verdict:
  - PASS -> recognizable Urdu -> proceed to default swap.
  - FAIL -> stop and re-research alternatives (no default swap).

## Default Engine Swap

Only after verdict PASS:

- `ingestor.py` default `ocr_engine` -> `UTRNetOCREngine()`.
- `PaddleOCREngine` stays as constructor-injectable fallback (unchanged).
- `ocr_lang` metadata remains `"ur"`.
- No `EasyOCREngine` is implemented.
- Swap is the LAST code change, made only on PASS.

## Testing

Unit tests (no models needed):

- `converter.py`: CTCLabelConverter encode/decode round-trip; NormalizePAD
  output shape `(1, 32, 400)`.
- `recognizer.py` / `detector.py`: preprocess pipeline shape using a dummy
  tensor / fake YOLO model.
- `urdu_ocr_engine.py`: contract tests with fake detector + fake recognizer —
  `lang` attribute, lazy imports, join-with-`"\n"`, `""` for no lines, clear
  missing-model error.

Integration tests (auto-skip when `models/` or weights absent):

- Real `UTRNetOCREngine.extract_text` on `images.jpg` or a generated Urdu PIL
  image, reusing the existing `test_engines_integration.py` auto-skip pattern.

Validation gate stays a manual `tools/validate_ocr.py` run (not in pytest).

Full-suite gate: `.\.venv\Scripts\python.exe -m pytest -v` must pass.

## Torch 2.13 / Py3.12 Compat Pass

- `torch.load`: torch 2.6+ defaults to `weights_only=True`; use explicit
  `map_location="cpu"` and `weights_only=False` if the checkpoint has non-tensor
  entries (verified at implementation time).
- `AdaptiveAvgPool2d((None, 1))` is fine on modern torch.
- PIL modern API (`Image.Transpose`, `Image.Resampling`) already used upstream.
- Verify ultralytics + numpy/opencv version compatibility at install time.
- CPU inference will be slow-ish but acceptable for a local pipeline (EasyOCR
  similarly took 30s-2min on this image).

## Sequence of Work

1. Vendor `ingestion/urdu_ocr/` package (adapt + compat pass + NOTICE.md).
2. Add `tools/download_urdu_models.py`; download weights into gitignored
   `models/`.
3. Extend `tools/validate_ocr.py` with the UTRNet candidate; run on
   `images.jpg`; record verdict in `.superpowers/sdd/ocr-validation.md`.
4. Unit + integration tests.
5. (PASS only) Flip the default engine in `ingestor.py` to UTRNet.
6. Update `requirements.txt` (add ultralytics, remove easyocr).
7. Run full suite; commit.

## References

- UTRNet paper: Rahman, Ghosh, Arora. ICDAR 2023, doi:10.1007/978-3-031-41734-4_19
- Upstream repo: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
- Official webapp: https://github.com/abdur75648/End-To-End-Urdu-OCR-WebApp
- Weights host: https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
