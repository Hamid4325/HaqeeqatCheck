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
