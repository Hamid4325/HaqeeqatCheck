import pytest

from ingestion.cascade_ocr import CascadeOCREngine


class FakeEngine:
    def __init__(self, lines):
        self.lines = lines
        self.calls = 0
        self.name = type(self).__name__

    def extract_lines(self, image_path):
        self.calls += 1
        return self.lines


class FailingEngine:
    name = "FailingEngine"

    def extract_lines(self, image_path):
        raise RuntimeError("model missing")


def test_lang_is_auto():
    assert CascadeOCREngine().lang == "auto"


def test_default_engines_are_urdu_primary_and_english_fallback():
    from ingestion.paddle_ocr_engine import PaddleOCREngine
    from ingestion.urdu_ocr import UTRNetOCREngine

    engine = CascadeOCREngine()
    assert isinstance(engine._primary, UTRNetOCREngine)
    assert isinstance(engine._fallback, PaddleOCREngine)
    assert engine._fallback.lang == "en"


def test_uses_primary_when_confidence_high():
    primary = FakeEngine([("سلام دنیا", 0.95)])
    fallback = FakeEngine([("hello world", 0.99)])
    engine = CascadeOCREngine(primary=primary, fallback=fallback)
    assert engine.extract_text("x.png") == "سلام دنیا"
    assert engine.engine_used == "FakeEngine"
    assert fallback.calls == 0


def test_uses_fallback_when_primary_low_confidence():
    primary = FakeEngine([("علب", 0.3)])
    fallback = FakeEngine([("hello world", 0.98)])
    engine = CascadeOCREngine(primary=primary, fallback=fallback)
    assert engine.extract_text("x.png") == "hello world"
    assert engine.engine_used == "FakeEngine"
    assert primary.calls == 1
    assert fallback.calls == 1


def test_uses_fallback_when_primary_empty():
    primary = FakeEngine([])
    fallback = FakeEngine([("hello world", 0.9)])
    engine = CascadeOCREngine(primary=primary, fallback=fallback)
    assert engine.extract_text("x.png") == "hello world"
    assert engine.engine_used == "FakeEngine"


def test_uses_fallback_when_primary_errors():
    fallback = FakeEngine([("hello world", 0.9)])
    engine = CascadeOCREngine(primary=FailingEngine(), fallback=fallback)
    assert engine.extract_text("x.png") == "hello world"
    assert engine.engine_used == "FakeEngine"


def test_joins_multiple_lines():
    primary = FakeEngine([("شکریہ", 0.9), ("پاکستان", 0.95)])
    engine = CascadeOCREngine(primary=primary, fallback=FakeEngine([]))
    assert engine.extract_text("x.png") == "شکریہ\nپاکستان"


def test_empty_when_primary_empty_and_fallback_errors():
    engine = CascadeOCREngine(primary=FakeEngine([]), fallback=FailingEngine())
    with pytest.raises(RuntimeError):
        engine.extract_text("x.png")


def test_extract_lines_passthrough():
    primary = FakeEngine([("سلام دنیا", 0.92)])
    engine = CascadeOCREngine(primary=primary, fallback=FakeEngine([]))
    assert engine.extract_lines("x.png") == [("سلام دنیا", 0.92)]


def test_falls_back_when_primary_confident_but_latin_script():
    primary = FakeEngine([("Spأو 595 5 2c8", 0.94)])
    fallback = FakeEngine([("hello world", 0.98)])
    engine = CascadeOCREngine(primary=primary, fallback=fallback)
    assert engine.extract_text("x.png") == "hello world"
    assert engine.engine_used == "FakeEngine"


def test_keeps_primary_when_confident_and_urdu_script():
    primary = FakeEngine([("شکریہ پاکستان", 0.9)])
    fallback = FakeEngine([("hello world", 0.98)])
    engine = CascadeOCREngine(primary=primary, fallback=fallback)
    assert engine.extract_text("x.png") == "شکریہ پاکستان"
    assert engine.engine_used == "FakeEngine"
    assert fallback.calls == 0
