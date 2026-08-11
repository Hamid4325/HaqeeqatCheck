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
