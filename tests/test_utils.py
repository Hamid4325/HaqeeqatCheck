import tempfile
from pathlib import Path

import pytest

from ingestion import utils

URDU = "\u0627\u0644\u0633\u0644\u0627\u0645 \u0639\u0644\u06cc\u06a9\u0645"  # Ø§Ù„Ø³Ù„Ø§Ù… Ø¹Ù„ÛŒÚ©Ù…


class TestTempDirManager:
    def test_creates_and_cleans_up(self):
        with utils.TempDirManager() as tmp:
            probe = Path(tmp) / "probe.txt"
            probe.write_text("x", encoding="utf-8")
            assert probe.exists()
            assert str(probe).startswith(tempfile.gettempdir())
        assert not probe.exists()

    def test_cleanup_on_exception(self):
        with pytest.raises(RuntimeError):
            with utils.TempDirManager() as tmp:
                probe = Path(tmp) / "probe.txt"
                probe.write_text("x", encoding="utf-8")
                raise RuntimeError("boom")
        assert not probe.exists()


class TestDeduplicateText:
    def test_keeps_first_occurrence(self):
        lines = ["salam duniya", "SALAM   duniya", "hello", ""]
        assert utils.deduplicate_text(lines) == ["salam duniya", "hello"]

    def test_empty_input(self):
        assert utils.deduplicate_text([]) == []


class TestBidi:
    def test_combine_both_sections(self):
        combined = utils.bidi_safe_combine("hello world", URDU)
        assert combined == "[AUDIO]: hello world\n[SCREEN TEXT]: " + utils.RLI + URDU + utils.PDI

    def test_combine_empty_ocr(self):
        assert utils.bidi_safe_combine("only audio", "") == "[AUDIO]: only audio"

    def test_combine_empty_audio(self):
        assert utils.bidi_safe_combine("", URDU) == "[SCREEN TEXT]: " + utils.RLI + URDU + utils.PDI

    def test_strip_bidi_marks(self):
        text = "[SCREEN TEXT]: " + utils.RLI + URDU + utils.PDI
        assert utils.strip_bidi_marks(text) == "[SCREEN TEXT]: " + URDU

    def test_ascii_only_is_not_wrapped(self):
        assert utils.bidi_safe_combine("a", "b") == "[AUDIO]: a\n[SCREEN TEXT]: b"


class TestVideoSampling:
    def test_samples_one_frame_per_interval(self, tmp_media, tmp_path):
        video_path = tmp_media.video(seconds=2, fps=15)  # 30 frames
        out_dir = tmp_path / "frames"
        out_dir.mkdir()
        frames = utils.sample_video_frames(video_path, str(out_dir), interval_sec=1)
        assert len(frames) == 2  # t=0 and t=1
        assert all(Path(f).exists() for f in frames)

    def test_duration(self, tmp_media):
        video_path = tmp_media.video(seconds=2, fps=15)
        assert utils.video_duration_sec(video_path) == pytest.approx(2.0, abs=0.1)
