import sys
import types

from ingestion.paddle_ocr_engine import PaddleOCREngine

URDU = "\u0633\u0644\u0627\u0645 \u062f\u0646\u06cc\u0627"  # "سلام دنیا"


def test_extract_text_parses_result(monkeypatch, tmp_path):
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"jpg")

    class FakeOCR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def ocr(self, image_path, cls=True):
            return [[
                [[0, 0, 1, 1], ("\u0633\u0644\u0627\u0645", 0.98)],
                [[0, 0, 1, 1], ("\u062f\u0646\u06cc\u0627", 0.95)],
            ]]

    fake_paddleocr = types.ModuleType("paddleocr")
    fake_paddleocr.PaddleOCR = FakeOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)

    engine = PaddleOCREngine()
    assert engine.extract_text(str(img)) == "\u0633\u0644\u0627\u0645\n\u062f\u0646\u06cc\u0627"
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


def test_constructor_falls_back_on_unknown_argument(monkeypatch, tmp_path):
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"jpg")

    class FakeOCR:
        def __init__(self, **kwargs):
            if "use_textline_orientation" in kwargs:
                raise ValueError("Unknown argument: use_textline_orientation")
            self.kwargs = kwargs

        def ocr(self, image_path, cls=True):
            return [[[[0, 0, 1, 1], ("\u0633\u0644\u0627\u0645", 0.9)]]]

    fake_paddleocr = types.ModuleType("paddleocr")
    fake_paddleocr.PaddleOCR = FakeOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)

    engine = PaddleOCREngine()
    assert engine.extract_text(str(img)) == "\u0633\u0644\u0627\u0645"
    assert engine._ocr.kwargs.get("lang") == "ur"


def test_constructor_falls_back_on_type_error(monkeypatch, tmp_path):
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"jpg")

    class FakeOCR:
        def __init__(self, **kwargs):
            if "use_textline_orientation" in kwargs:
                raise TypeError("unexpected keyword argument")
            self.kwargs = kwargs

        def ocr(self, image_path, cls=True):
            return [[[[0, 0, 1, 1], ("\u0633\u0644\u0627\u0645", 0.9)]]]

    fake_paddleocr = types.ModuleType("paddleocr")
    fake_paddleocr.PaddleOCR = FakeOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)

    engine = PaddleOCREngine()
    assert engine.extract_text(str(img)) == "\u0633\u0644\u0627\u0645"
    assert engine._ocr.kwargs.get("lang") == "ur"


def test_extract_text_falls_back_on_deprecated_cls_arg(monkeypatch, tmp_path):
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"jpg")

    class FakeOCR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def ocr(self, image_path, cls=None):
            if cls is not None:
                raise TypeError("cls is deprecated")
            return [[[[0, 0, 1, 1], ("\u0633\u0644\u0627\u0645", 0.9)]]]

    fake_paddleocr = types.ModuleType("paddleocr")
    fake_paddleocr.PaddleOCR = FakeOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)

    assert PaddleOCREngine().extract_text(str(img)) == "\u0633\u0644\u0627\u0645"


def test_extract_lines_returns_confidence_pairs(monkeypatch, tmp_path):
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"jpg")

    class FakeOCR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def ocr(self, image_path, cls=True):
            return [[
                [[0, 0, 1, 1], ("\u0633\u0644\u0627\u0645", 0.98)],
                [[0, 0, 1, 1], ("\u062f\u0646\u06cc\u0627", 0.91)],
            ]]

    fake_paddleocr = types.ModuleType("paddleocr")
    fake_paddleocr.PaddleOCR = FakeOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)

    engine = PaddleOCREngine()
    assert engine.extract_lines(str(img)) == [
        ("\u0633\u0644\u0627\u0645", 0.98),
        ("\u062f\u0646\u06cc\u0627", 0.91),
    ]


def test_extract_lines_empty_when_none_pages(monkeypatch, tmp_path):
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

    assert PaddleOCREngine().extract_lines(str(img)) == []


def test_constructor_sets_text_det_limit_side_len(monkeypatch, tmp_path):
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"jpg")

    class FakeOCR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def ocr(self, image_path, cls=True):
            return [[[[0, 0, 1, 1], ("hi", 0.9)]]]

    fake_paddleocr = types.ModuleType("paddleocr")
    fake_paddleocr.PaddleOCR = FakeOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)

    engine = PaddleOCREngine()
    engine.extract_text(str(img))
    assert engine._ocr.kwargs.get("text_det_limit_side_len") == 640


def test_fallback_branch_keeps_text_det_limit_side_len(monkeypatch, tmp_path):
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"jpg")

    class FakeOCR:
        def __init__(self, **kwargs):
            if "use_textline_orientation" in kwargs:
                raise ValueError("Unknown argument: use_textline_orientation")
            self.kwargs = kwargs

        def ocr(self, image_path, cls=True):
            return [[[[0, 0, 1, 1], ("hi", 0.9)]]]

    fake_paddleocr = types.ModuleType("paddleocr")
    fake_paddleocr.PaddleOCR = FakeOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)

    engine = PaddleOCREngine()
    engine.extract_text(str(img))
    assert engine._ocr.kwargs.get("text_det_limit_side_len") == 640


def test_parse_handles_ocr_v3_dict_result(monkeypatch, tmp_path):
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"jpg")

    class FakeOCR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def ocr(self, image_path, cls=True):
            return [{"rec_texts": ["hello", "world"], "rec_scores": [0.99, 0.85]}]

    fake_paddleocr = types.ModuleType("paddleocr")
    fake_paddleocr.PaddleOCR = FakeOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)

    engine = PaddleOCREngine()
    assert engine.extract_lines(str(img)) == [("hello", 0.99), ("world", 0.85)]
    assert engine.extract_text(str(img)) == "hello\nworld"
