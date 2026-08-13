import tempfile
from pathlib import Path

import pytest

from ingestion.base import UnsupportedFormatError
from ingestion.ingestor import HaqeeqatIngestor

URDU_OCR = "\u0633\u0644\u0627\u0645 \u062f\u0646\u06cc\u0627"  # Ø³Ù„Ø§Ù… Ø¯Ù†ÛŒØ§


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

    def __init__(self, result=URDU_OCR, engine_used="FakeOCREngine"):
        self.result = result
        self.called_with = []
        self.engine_used = engine_used

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

    def test_metadata_reports_engine_used(self, tmp_media):
        report = make_ingestor().ingest(tmp_media.image())
        assert report["metadata"]["ocr_engine_used"] == "FakeOCREngine"

    def test_default_ocr_engine_is_cascade(self):
        from ingestion.cascade_ocr import CascadeOCREngine

        ingestor = HaqeeqatIngestor(FakeTranscriber())
        assert isinstance(ingestor.ocr_engine, CascadeOCREngine)
