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
    def __init__(self, result, confidence=0.9):
        self.result = result
        self.confidence = confidence
        self.calls = 0

    def recognize(self, crop):
        self.calls += 1
        return self.result

    def recognize_with_confidence(self, crop):
        self.calls += 1
        return self.result, self.confidence


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


def test_extract_lines_returns_confidence_pairs(sample_image):
    engine = UTRNetOCREngine(
        detector=FakeDetector([Image.new("RGB", (10, 10))]),
        recognizer=FakeRecognizer("شکریہ پاکستان", confidence=0.87),
    )
    assert engine.extract_lines(sample_image) == [("شکریہ پاکستان", 0.87)]


def test_extract_lines_skips_blank_lines(sample_image):
    recognizer = FakeRecognizer("   ", confidence=0.1)
    engine = UTRNetOCREngine(detector=FakeDetector([Image.new("RGB", (10, 10))]), recognizer=recognizer)
    assert engine.extract_lines(sample_image) == []


def test_recognizer_confidence_uses_mean_softmax(tmp_path, monkeypatch):
    import torch

    from ingestion.urdu_ocr.recognizer import UTRNetRecognizer

    class FakeModel:
        def __call__(self, tensor):
            logits = torch.zeros(1, 3, 4)
            logits[0, 0, 1] = 10.0
            logits[0, 1, 2] = 10.0
            logits[0, 2, 3] = 10.0
            return logits

    class FakeConverter:
        character = ["[CTCblank]", "a", "b", "c"]

        def decode(self, text_index, length):
            return ["abc"]

    rec = UTRNetRecognizer("dummy.pth")
    rec._model = FakeModel()
    rec._converter = FakeConverter()
    text, conf = rec.recognize_with_confidence(Image.new("L", (60, 32), 255))
    assert text == "abc"
    assert conf == pytest.approx(1.0, abs=1e-3)


def test_recognizer_confidence_drops_when_logits_flat(tmp_path):
    import torch

    from ingestion.urdu_ocr.recognizer import UTRNetRecognizer

    class FakeModel:
        def __call__(self, tensor):
            return torch.zeros(1, 3, 4)

    class FakeConverter:
        character = ["[CTCblank]", "a", "b", "c"]

        def decode(self, text_index, length):
            return [""]

    rec = UTRNetRecognizer("dummy.pth")
    rec._model = FakeModel()
    rec._converter = FakeConverter()
    _, conf = rec.recognize_with_confidence(Image.new("L", (60, 32), 255))
    assert conf == pytest.approx(0.25, abs=1e-3)


def test_recognizer_recognize_still_returns_plain_text(tmp_path):
    import torch

    from ingestion.urdu_ocr.recognizer import UTRNetRecognizer

    class FakeModel:
        def __call__(self, tensor):
            logits = torch.zeros(1, 3, 4)
            logits[0, 0, 1] = 10.0
            logits[0, 1, 2] = 10.0
            logits[0, 2, 3] = 10.0
            return logits

    class FakeConverter:
        character = ["[CTCblank]", "a", "b", "c"]

        def decode(self, text_index, length):
            return ["abc"]

    rec = UTRNetRecognizer("dummy.pth")
    rec._model = FakeModel()
    rec._converter = FakeConverter()
    assert rec.recognize(Image.new("L", (60, 32), 255)) == "abc"
