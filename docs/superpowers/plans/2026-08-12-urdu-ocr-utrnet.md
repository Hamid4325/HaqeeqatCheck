# UTRNet + YOLOv8 Urdu OCR Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vendor the official UTRNet end-to-end Urdu OCR pipeline (YOLOv8 line detection + UTRNet recognition) as a new `OCR` engine, validate it against `images.jpg` (expected شکریہ پاکستان), and make it the default OCR engine in `HaqeeqatIngestor`.

**Architecture:** New package `ingestion/urdu_ocr/` with composable units — `converter.py` (CTC + NormalizePAD), `unet.py`, `model.py` (UNet + BiLSTM + Linear), `recognizer.py`, `detector.py` (YOLOv8), composed by thin `UTRNetOCREngine(OCREngine)`. Weights (~91 MB) download into a gitignored `models/`. `tools/validate_ocr.py` extended to test UTRNet; the ingestor default flips to UTRNet only on PASS.

**Tech Stack:** Python 3.12, torch 2.13.0+cpu (installed), ultralytics (to add), Pillow (installed), pytest.

## Global Constraints

- `OCREngine.extract_text(image_path: str) -> str` contract unchanged.
- Heavy imports (torch, ultralytics) lazy — inside methods only, never at module top level.
- All engines keep a `lang` attribute (read by `_base_metadata` in `ingestor.py`).
- No secrets. No committing model weights (`models/` gitignored).
- Vendored code stays architecturally faithful to upstream so `best_norm_ED.pth` loads as-is.
- Attribution for vendored code (CC BY-NC-SA 4.0) in `ingestion/urdu_ocr/NOTICE.md`.
- Full-suite gate: `.\.venv\Scripts\python.exe -m pytest -v` must pass.
- `.superpowers/` and `images.jpg`/`testcheck.py` at repo root are untracked scratch — do NOT commit.

---

### Task 1: Vendor the UTRNet package — converter, UNet, model, glyphs, NOTICE

**Files:**
- Create: `ingestion/urdu_ocr/__init__.py`
- Create: `ingestion/urdu_ocr/UrduGlyphs.txt` (180 lines, one glyph per line — copy byte-for-byte from upstream `End-To-End-Urdu-OCR-WebApp/UrduGlyphs.txt`)
- Create: `ingestion/urdu_ocr/NOTICE.md`
- Create: `ingestion/urdu_ocr/converter.py`
- Create: `ingestion/urdu_ocr/unet.py`
- Create: `ingestion/urdu_ocr/model.py`
- Test: `tests/test_urdu_ocr_converter.py`
- Test: `tests/test_urdu_ocr_model.py`

**Interfaces:**
- Consumes: `torch`, `torchvision`, `numpy` (already installed).
- Produces:
  - `ingestion.urdu_ocr.converter.load_urdu_glyphs() -> str` (glyph vocab + trailing space)
  - `ingestion.urdu_ocr.converter.CTCLabelConverter(character: str)` with `.encode(text, batch_max_length=25) -> (LongTensor, IntTensor)` and `.decode(text_index, length) -> list[str]`
  - `ingestion.urdu_ocr.converter.NormalizePAD(max_size: tuple)` — `__call__(PIL image) -> FloatTensor(1, 32, 400)`
  - `ingestion.urdu_ocr.model.Model(num_class=181, device="cpu")` — `.to(device)`, `.load_state_dict(...)`, `.eval()`, `.forward(input) -> FloatTensor(1, T, num_class)`
  - `ingestion.urdu_ocr.UNet_FeatureExtractor(input_channel=1, output_channel=512)` and `BidirectionalLSTM`, `DropoutLayer` (used internally by `Model`)

- [ ] **Step 1: Write the failing unit tests**

`tests/test_urdu_ocr_converter.py`:

```python
import pytest
from PIL import Image

from ingestion.urdu_ocr.converter import CTCLabelConverter, NormalizePAD, load_urdu_glyphs


def test_load_urdu_glyphs_has_vocab_and_space():
    glyphs = load_urdu_glyphs()
    assert glyphs.endswith(" ")
    assert "ا" in glyphs and "پ" in glyphs and "ش" in glyphs
    assert "A" in glyphs and "0" in glyphs


def test_ctc_converter_round_trip():
    glyphs = load_urdu_glyphs()
    converter = CTCLabelConverter(glyphs)
    indices, length = converter.encode(["شکریہ پاکستان"])
    decoded = converter.decode(indices, length)
    assert decoded == ["شکریہ پاکستان"]


def test_ctc_converter_removes_blanks_and_repeats():
    glyphs = load_urdu_glyphs()
    converter = CTCLabelConverter(glyphs)
    # [0] is the CTC blank token
    decoded = converter.decode([0, 0, 1, 1, 2], length=[5])
    assert decoded == ["ا" + "آ"]


def test_normalizepad_output_shape():
    img = Image.new("L", (60, 32), color=128)
    out = NormalizePAD((1, 32, 400))(img)
    assert tuple(out.shape) == (1, 32, 400)
```

`tests/test_urdu_ocr_model.py`:

```python
import pytest

torch = pytest.importorskip("torch")


def test_model_forward_shape():
    from ingestion.urdu_ocr.model import Model

    model = Model(num_class=10, device="cpu")
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 32, 400))
    assert tuple(out.shape) == (1, 400, 10)


def test_model_loads_state_dict_keys_match():
    from ingestion.urdu_ocr.model import Model

    model = Model(num_class=541, device="cpu")
    keys = set(model.state_dict().keys())
    assert "FeatureExtraction.ConvNet.inc.double_conv.0.weight" in keys
    assert "SequenceModeling.0.rnn.weight_ih_l0" in keys
    assert "Prediction.weight" in keys
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_urdu_ocr_converter.py tests/test_urdu_ocr_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.urdu_ocr'`

- [ ] **Step 3: Create `ingestion/urdu_ocr/UrduGlyphs.txt`**

Verbatim from upstream (`End-To-End-Urdu-OCR-WebApp`). 180 lines, each a single glyph (see the temp copy at `C:\Users\hp\AppData\Local\Temp\opencode\utrnet-src\UrduGlyphs.txt`). First 9 lines: `ا`, `آ`, `ب`, `پ`, `ت`, `ٹ`, `ث`, `ج`, `چ`.

- [ ] **Step 4: Create `ingestion/urdu_ocr/NOTICE.md`**

```markdown
# NOTICE

The files in this directory are adapted from the UTRNet end-to-end Urdu OCR
web app by Abdur Rahman, Arjun Ghosh, and Chetan Arora:

- https://github.com/abdur75648/End-To-End-Urdu-OCR-WebApp
- UTRNet paper: Rahman, Ghosh, Arora. ICDAR 2023, doi:10.1007/978-3-031-41734-4_19

Adapted: converter.py, unet.py, model.py, recognizer.py, detector.py,
urdu_ocr_engine.py, UrduGlyphs.txt.

License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0). Non-commercial use only.
Model weights: best_norm_ED.pth, yolov8m_UrduDoc.pt (hosted by the authors
at https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet).
```

- [ ] **Step 5: Create `ingestion/urdu_ocr/converter.py`**

```python
"""CTC label converter and image normalization (adapted from UTRNet webapp utils.py)."""

import math
from importlib import resources

import torch
import torchvision.transforms as T


def load_urdu_glyphs() -> str:
    """Return the UTRNet Urdu glyph vocabulary as a single string plus trailing space."""
    text = (
        resources.files("ingestion.urdu_ocr")
        .joinpath("UrduGlyphs.txt")
        .read_text(encoding="utf-8")
    )
    glyphs = "".join(line for line in text.splitlines())
    return glyphs + " "


class NormalizePAD:
    """Convert a PIL image to a normalized, right-padded FloatTensor of max_size."""

    def __init__(self, max_size, PAD_type="right"):
        self.toTensor = T.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        pad_img[:, :, :w] = img
        if self.max_size[2] != w:
            pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return pad_img


class CTCLabelConverter:
    """Convert between text labels and CTC indices. Index 0 is the CTC blank."""

    def __init__(self, character):
        dict_character = list(character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1
        self.character = ["[CTCblank]"] + dict_character

    def encode(self, text, batch_max_length=25):
        length = [len(s) for s in text]
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            indices = [self.dict[char] for char in t]
            batch_text[i][: len(indices)] = torch.LongTensor(indices)
        return batch_text, torch.IntTensor(length)

    def decode(self, text_index, length):
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character[t[i]])
            texts.append("".join(char_list))
        return texts
```

Note: `resources.files("ingestion.urdu_ocr")` requires `ingestion/__init__.py` to import fine without heavy deps — it does.

- [ ] **Step 6: Create `ingestion/urdu_ocr/unet.py`**

```python
"""UNet backbone used by the UTRNet feature extractor (adapted from webapp modules/cnn/unet.py)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=512):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
```

- [ ] **Step 7: Create `ingestion/urdu_ocr/model.py`**

```python
"""UTRNet recognition model (adapted from webapp model.py + modules/)."""

import numpy as np
import torch
import torch.nn as nn

from ingestion.urdu_ocr.unet import UNet


class UNet_FeatureExtractor(nn.Module):
    def __init__(self, input_channel=1, output_channel=512):
        super().__init__()
        self.ConvNet = UNet(input_channel, output_channel)

    def forward(self, input):
        return self.ConvNet(input)


class DropoutLayer(nn.Module):
    """Feature-wise dropout mask (random per timestep, upstream keeps it at inference)."""

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, input):
        nums = (np.random.rand(input.shape[1]) > 0.2).astype(int)
        mask = torch.from_numpy(nums).to(self.device)
        mask = torch.reshape(mask, (input.shape[1], 1)).to(self.device)
        mask = mask.repeat(input.shape[0], 1, input.shape[2]).to(self.device)
        return input * mask


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        return self.linear(recurrent)


class Model(nn.Module):
    """UNet extractor + temporal dropout + 2x BiLSTM + linear prediction."""

    def __init__(self, num_class=181, device="cpu"):
        super().__init__()
        self.device = device
        self.FeatureExtraction = UNet_FeatureExtractor(1, 512)
        self.FeatureExtraction_output = 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.dropout1 = DropoutLayer(self.device)
        self.dropout2 = DropoutLayer(self.device)
        self.dropout3 = DropoutLayer(self.device)
        self.dropout4 = DropoutLayer(self.device)
        self.dropout5 = DropoutLayer(self.device)
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, 256, 256),
            BidirectionalLSTM(256, 256, 256),
        )
        self.SequenceModeling_output = 256
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    def forward(self, input, text=None, is_train=True):
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)
        branches = [
            self.SequenceModeling(dropout(visual_feature))
            for dropout in (self.dropout1, self.dropout2, self.dropout3, self.dropout4, self.dropout5)
        ]
        contextual_feature = branches[0]
        for branch in branches[1:]:
            contextual_feature = contextual_feature.add(branch)
        contextual_feature = contextual_feature * (1 / 5)
        return self.Prediction(contextual_feature.contiguous())
```

- [ ] **Step 8: Create `ingestion/urdu_ocr/__init__.py`**

Use a lazy module-level `__getattr__` (PEP 562) so `import ingestion.urdu_ocr`
does NOT pull in torch/ultralytics at import time (matches the repo's
lazy-heavy-import convention):

```python
"""UTRNet + YOLOv8 end-to-end Urdu OCR package (lazy exports).

Heavy imports (torch, ultralytics) happen only when the engine is first used,
so importing this package stays fast.
"""

__all__ = ["UTRNetOCREngine"]


def __getattr__(name):
    if name == "UTRNetOCREngine":
        from ingestion.urdu_ocr.urdu_ocr_engine import UTRNetOCREngine

        return UTRNetOCREngine
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
```

(Note: `urdu_ocr_engine.py` is created in Task 2 — `from ingestion.urdu_ocr import UTRNetOCREngine` only resolves after Task 2. Tests in Task 1 import `converter`/`model` submodules directly, so they pass first.)

- [ ] **Step 9: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_urdu_ocr_converter.py tests/test_urdu_ocr_model.py -v`
Expected: 7 PASSED. (The model tests run the UNet forward on CPU — takes a few seconds.)

- [ ] **Step 10: Commit**

```bash
git add ingestion/urdu_ocr tests/test_urdu_ocr_converter.py tests/test_urdu_ocr_model.py
git commit -m "feat: vendor UTRNet model, converter, and Urdu glyph vocab"
```

---

### Task 2: Recognizer, detector, and UTRNetOCREngine

**Files:**
- Create: `ingestion/urdu_ocr/recognizer.py`
- Create: `ingestion/urdu_ocr/detector.py`
- Create: `ingestion/urdu_ocr/urdu_ocr_engine.py`
- Test: `tests/test_urdu_ocr_engine.py`

**Interfaces:**
- Consumes: `converter.CTCLabelConverter`, `converter.NormalizePAD`, `converter.load_urdu_glyphs`, `model.Model`, `base.OCREngine` (Task 1).
- Produces:
  - `ingestion.urdu_ocr.recognizer.UTRNetRecognizer(checkpoint_path, device="cpu")` with `.recognize(pil_crop) -> str`
  - `ingestion.urdu_ocr.detector.TextLineDetector(weights_path, device="cpu")` with `.detect(pil_image) -> list[PIL.Image]`
  - `ingestion.urdu_ocr.urdu_ocr_engine.UTRNetOCREngine(OCREngine)` with `.extract_text(image_path: str) -> str`
  - Exported from package `__init__` as `UTRNetOCREngine`

- [ ] **Step 1: Write the failing unit tests**

`tests/test_urdu_ocr_engine.py`:

```python
import os

import pytest
from PIL import Image

from ingestion.urdu_ocr.urdu_ocr_engine import UTRNetOCREngine


class FakeDetector:
    def __init__(self, crops):
        self.crops = crops
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return self.crops


class FakeRecognizer:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def recognize(self, crop):
        self.calls += 1
        return self.result


@pytest.fixture
def sample_image(tmp_path):
    img = Image.new("RGB", (60, 30), color="white")
    path = tmp_path / "sample.png"
    img.save(path)
    return str(path)


def test_engine_lang_defaults_to_ur():
    assert UTRNetOCREngine().lang == "ur"


def test_engine_constructor_is_lazy():
    engine = UTRNetOCREngine()
    assert engine._recognizer is None
    assert engine._detector is None


def test_extract_text_joins_lines(sample_image):
    engine = UTRNetOCREngine(
        detector=FakeDetector([Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10))]),
        recognizer=FakeRecognizer("شکریہ پاکستان"),
    )
    assert engine.extract_text(sample_image) == "شکریہ پاکستان\nشکریہ پاکستان"


def test_extract_text_empty_when_no_lines(sample_image):
    recognizer = FakeRecognizer("x")
    engine = UTRNetOCREngine(detector=FakeDetector([]), recognizer=recognizer)
    assert engine.extract_text(sample_image) == ""
    assert recognizer.calls == 0


def test_extract_text_empty_on_unreadable_image(tmp_path):
    bogus = tmp_path / "bogus.jpg"
    bogus.write_bytes(b"not an image")
    engine = UTRNetOCREngine(
        detector=FakeDetector([Image.new("RGB", (10, 10))]),
        recognizer=FakeRecognizer("x"),
    )
    assert engine.extract_text(str(bogus)) == ""


def test_missing_models_raise_clear_error(tmp_path):
    engine = UTRNetOCREngine(model_dir=str(tmp_path), device="cpu")
    with pytest.raises(FileNotFoundError, match="download_urdu_models"):
        engine.extract_text(str(tmp_path / "whatever.png"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_urdu_ocr_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.urdu_ocr.urdu_ocr_engine'`

- [ ] **Step 3: Create `ingestion/urdu_ocr/recognizer.py`**

```python
"""Single text-line recognizer using the UTRNet model (adapted from webapp read.py)."""

import math

import torch
from PIL import Image

from ingestion.urdu_ocr.converter import CTCLabelConverter, NormalizePAD, load_urdu_glyphs
from ingestion.urdu_ocr.model import Model


def preprocess(pil_crop) -> torch.Tensor:
    """Grayscale -> mirror -> resize (h=32, w<=400) -> NormalizePAD(1,32,400)."""
    img = pil_crop.convert("L")
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = img.size
    ratio = w / float(h)
    resized_w = 400 if math.ceil(32 * ratio) > 400 else math.ceil(32 * ratio)
    img = img.resize((resized_w, 32), Image.Resampling.BICUBIC)
    return NormalizePAD((1, 32, 400))(img)


class UTRNetRecognizer:
    """Recognizes the text in one cropped line image."""

    def __init__(self, checkpoint_path, device="cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None
        self._converter = None

    def _load(self):
        if self._model is not None:
            return
        glyphs = load_urdu_glyphs()
        self._converter = CTCLabelConverter(glyphs)
        model = Model(num_class=len(self._converter.character), device=self.device)
        state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self._model = model

    def recognize(self, pil_crop) -> str:
        self._load()
        tensor = preprocess(pil_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self._model(tensor)
        preds_size = torch.IntTensor([preds.size(1)])
        _, preds_index = preds.max(2)
        return self._converter.decode(preds_index.data, preds_size.data)[0]
```

- [ ] **Step 4: Create `ingestion/urdu_ocr/detector.py`**

```python
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
```

- [ ] **Step 5: Create `ingestion/urdu_ocr/urdu_ocr_engine.py`**

```python
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
        self._load_models()
        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
        except OSError:
            return ""
        crops = self._detector.detect(image)
        lines = []
        for crop in crops:
            text = self._recognizer.recognize(crop)
            if text.strip():
                lines.append(text.strip())
        return "\n".join(lines)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_urdu_ocr_engine.py -v`
Expected: 6 PASSED

- [ ] **Step 7: Commit**

```bash
git add ingestion/urdu_ocr/recognizer.py ingestion/urdu_ocr/detector.py ingestion/urdu_ocr/urdu_ocr_engine.py tests/test_urdu_ocr_engine.py
git commit -m "feat: add UTRNetOCR engine (YOLOv8 detection + UTRNet recognition)"
```

---

### Task 3: Model download script + gitignored models/

**Files:**
- Create: `tools/download_urdu_models.py`
- Modify: `.gitignore` (append `models/`)

**Interfaces:**
- Consumes: network access to `https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet/resolve/main/`.
- Produces: `models/best_norm_ED.pth` (41 MB) and `models/yolov8m_UrduDoc.pt` (49.7 MB) at repo root.

- [ ] **Step 1: Create `tools/download_urdu_models.py`**

```python
"""Download UTRNet + YOLOv8-UrduDoc weights into the gitignored models/ dir.

Usage:
    python tools/download_urdu_models.py [--force]
"""

import argparse
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
BASE_URL = "https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet/resolve/main"
FILES = {
    "best_norm_ED.pth": 41 * 1024 * 1024,
    "yolov8m_UrduDoc.pt": int(49.7 * 1024 * 1024),
}


def download(name, expected_bytes, force):
    dest = os.path.join(MODELS_DIR, name)
    if os.path.isfile(dest) and os.path.getsize(dest) >= expected_bytes * 0.95 and not force:
        print("skip  %s (already present, %.1f MB)" % (name, os.path.getsize(dest) / 1048576))
        return
    url = "%s/%s" % (BASE_URL, name)
    print("fetch %s ..." % url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        total = 0
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
            if total % (2 * 1048576) < 65536:
                print("  %.1f MB ..." % (total / 1048576))
    print("saved %s (%.1f MB)" % (name, os.path.getsize(dest) / 1048576))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, size in FILES.items():
        download(name, size, args.force)
    print("Done. Models are in %s" % MODELS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Append `models/` to `.gitignore`**

`.gitignore` becomes:

```
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.paddleocr/
*.egg-info/
dist/
build/
.env
models/
```

- [ ] **Step 3: Run the download script**

Run: `.\.venv\Scripts\python.exe tools/download_urdu_models.py`
Expected: both files downloaded (may take minutes — use a long timeout). Then verify:
`.\.venv\Scripts\python.exe -c "import os; [print(f, round(os.path.getsize(os.path.join('models', f))/1048576,1), 'MB') for f in ('best_norm_ED.pth','yolov8m_UrduDoc.pt')]"`

- [ ] **Step 4: Commit**

```bash
git add tools/download_urdu_models.py .gitignore
git commit -m "feat: add UTRNet/YOLO model download script; ignore models/"
```

---

### Task 4: Extend the validation harness and run the gate

**Files:**
- Modify: `tools/validate_ocr.py`
- Output (NOT committed): `.superpowers/sdd/ocr-validation.md`

**Interfaces:**
- Consumes: `ingestion.urdu_ocr.UTRNetOCREngine` (Task 2), `models/` (Task 3), `images.jpg`.
- Produces: UTRNet result recorded next to PaddleOCR/EasyOCR with a PASS/FAIL verdict.

- [ ] **Step 1: Add a UTRNet block to `tools/validate_ocr.py`**

Inside `main()`, after the EasyOCR block, add:

```python
        record("engine: UTRNet (YOLOv8 + UTRNet)")
        try:
            from ingestion.urdu_ocr import UTRNetOCREngine

            engine = UTRNetOCREngine()
            utrnet_text = engine.extract_text(image_path)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            utrnet_text = "ERROR: %s: %s" % (type(exc).__name__, exc)
            exit_code = 1
        record("detected: %r" % utrnet_text)
```

Then, at the end of the per-image loop (inside the `for` over images), after the UTRNet record for `index == 0`, append the verdict:

```python
        if index == 0:
            matched = any(word in utrnet_text for word in ("شکریہ", "پاکستان"))
            if "ERROR" in utrnet_text:
                record("verdict: FAIL - UTRNet errored (see above); no default engine swap")
            elif matched:
                record("verdict: PASS - UTRNet recognized the expected Urdu text; proceed to default swap")
            else:
                record("verdict: FAIL - UTRNet returned %r, no match for %r; re-research" % (utrnet_text, EXPECTED_FIRST))
```

- [ ] **Step 2: Run the harness on `images.jpg`**

Run: `.\.venv\Scripts\python.exe tools/validate_ocr.py images.jpg` (long timeout — CPU inference)
Expected: UTRNet engine line prints; result appended to `.superpowers/sdd/ocr-validation.md`. Read that file to get the exact detected string.

- [ ] **Step 3: Record the verdict decision**

In the response, report the exact UTRNet-detected string for `images.jpg` and whether the gate PASSED or FAILED. If PASS, continue to Task 5. If FAIL, stop — no default swap; re-research alternative engines.

- [ ] **Step 4: Commit the harness change**

```bash
git add tools/validate_ocr.py
git commit -m "tools: add UTRNet candidate to OCR validation harness"
```

Note: `.superpowers/` stays untracked.

---

### Task 5: Integration test (auto-skip)

**Files:**
- Create: `tests/test_urdu_ocr_integration.py`

**Interfaces:**
- Consumes: `ingestion.urdu_ocr.UTRNetOCREngine`, `models/` weights, `images.jpg`.

- [ ] **Step 1: Write the test**

```python
import os

import pytest

from ingestion.urdu_ocr.urdu_ocr_engine import MODELS_DIR, RECOGNIZER_WEIGHTS, DETECTOR_WEIGHTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(
    not all(
        os.path.isfile(os.path.join(MODELS_DIR, name))
        for name in (RECOGNIZER_WEIGHTS, DETECTOR_WEIGHTS)
    ),
    reason="UTRNet/YOLO weights not downloaded (run tools/download_urdu_models.py)",
)
class TestUTRNetIntegration:
    def test_extract_text_returns_text(self):
        from ingestion.urdu_ocr import UTRNetOCREngine

        engine = UTRNetOCREngine()
        image = os.path.join(ROOT, "images.jpg")
        text = engine.extract_text(image)
        assert isinstance(text, str)
        assert len(text.strip()) > 0
```

- [ ] **Step 2: Run the suite to verify the auto-skip**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_urdu_ocr_integration.py -v`
Expected: SKIPPED if weights present... — run the full suite later. Actually with weights present it runs and asserts non-empty text.

- [ ] **Step 3: Commit**

```bash
git add tests/test_urdu_ocr_integration.py
git commit -m "test: add auto-skip integration test for UTRNet OCR engine"
```

---

### Task 6: Default engine swap + dependency cleanup + full suite

**Condition:** Only run this task if Task 4's verdict is PASS.

**Files:**
- Modify: `ingestion/ingestor.py` (import + default engine)
- Modify: `requirements.txt` (remove `easyocr`, add `ultralytics`)
- Test: existing `tests/test_ingestor.py`, `tests/test_main.py`

**Interfaces:**
- Consumes: `ingestion.urdu_ocr.UTRNetOCREngine` (Task 2).
- Produces: `HaqeeqatIngestor()` defaulting to `UTRNetOCREngine()`; `ocr_lang` metadata = `"ur"`.

- [ ] **Step 1: Swap the default engine in `ingestion/ingestor.py`**

Replace the import line:
```python
from ingestion.paddle_ocr_engine import PaddleOCREngine
```
with:
```python
from ingestion.urdu_ocr import UTRNetOCREngine
```

And in `__init__`:
```python
self.ocr_engine = ocr_engine or PaddleOCREngine()
```
becomes:
```python
self.ocr_engine = ocr_engine or UTRNetOCREngine()
```

- [ ] **Step 2: Update `requirements.txt`**

Remove the `easyocr` line. Append `ultralytics`.

Final contents:
```
openai-whisper
paddleocr
paddlepaddle
moviepy
opencv-python
numpy
ultralytics
```

- [ ] **Step 3: Install ultralytics**

Run: `.\.venv\Scripts\python.exe -m pip install ultralytics`
Expected: installs cleanly; confirm torch NOT reinstalled:
`.\.venv\Scripts\python.exe -m pip show torch | findstr Version` still `2.13.0+cpu` (or whatever was present).

- [ ] **Step 4: Run the full suite**

Run: `.\.venv\Scripts\python.exe -m pytest -v`
Expected: all tests pass (UTRNet integration test runs for real now; ~30-120s).

- [ ] **Step 5: Commit**

```bash
git add ingestion/ingestor.py requirements.txt
git commit -m "feat: make UTRNet the default OCR engine; swap easyocr for ultralytics"
```
