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
    import torch

    glyphs = load_urdu_glyphs()
    converter = CTCLabelConverter(glyphs)
    # [0] is the CTC blank token
    decoded = converter.decode(
        torch.LongTensor([[0, 0, 1, 1, 2]]), length=torch.IntTensor([5])
    )
    assert decoded == ["ا" + "آ"]


def test_normalizepad_output_shape():
    img = Image.new("L", (60, 32), color=128)
    out = NormalizePAD((1, 32, 400))(img)
    assert tuple(out.shape) == (1, 32, 400)
