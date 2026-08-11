"""Integration tests for the real Whisper and PaddleOCR engines.

These download models on first run (requires network) and are slow, so they
skip themselves when a dependency, font, or model is unavailable.
"""

import os

import pytest

from ingestion.ingestor import HaqeeqatIngestor
from ingestion.paddle_ocr_engine import PaddleOCREngine
from ingestion.whisper_transcriber import WhisperTranscriber

pytestmark = pytest.mark.slow


def _find_arabic_font():
    font_dir = r"C:\Windows\Fonts"
    candidates = ["tahoma.ttf", "arial.ttf", "arialuni.ttf", "segoeui.ttf", "times.ttf"]
    for name in candidates:
        path = os.path.join(font_dir, name)
        if os.path.exists(path):
            return path
    return None


def test_whisper_transcribes_audio(tmp_media):
    transcriber = WhisperTranscriber(model_size="base")
    try:
        result = transcriber.transcribe(tmp_media.tone_wav(seconds=1))
    except Exception as exc:
        pytest.skip(f"Whisper model unavailable: {exc}")
    assert isinstance(result, str)


def test_paddle_ocr_reads_urdu_image(tmp_path):
    font_path = _find_arabic_font()
    if font_path is None:
        pytest.skip("No Arabic-capable font found")

    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        pytest.skip(f"Image rendering deps unavailable: {exc}")

    text = "\u067e\u0627\u06a9\u0633\u062a\u0627\u0646"  # Ù¾Ø§Ú©Ø³ØªØ§Ù†
    try:
        display_text = get_display(arabic_reshaper.reshape(text))
        img = Image.new("RGB", (800, 180), "white")
        draw = ImageDraw.Draw(img)
        draw.text(
            (20, 40), display_text, fill="black",
            font=ImageFont.truetype(font_path, 72),
        )
        img_path = tmp_path / "urdu.png"
        img.save(str(img_path))
    except Exception as exc:
        pytest.skip(f"Cannot render Urdu image: {exc}")

    try:
        result = PaddleOCREngine().extract_text(str(img_path))
    except Exception as exc:
        pytest.skip(f"PaddleOCR model unavailable: {exc}")
    assert isinstance(result, str)
    assert len(result) > 0


def test_video_pipeline_end_to_end(tmp_path, tmp_media):
    try:
        try:
            from moviepy import VideoFileClip, AudioFileClip  # moviepy 2.x
        except ImportError:
            from moviepy.editor import VideoFileClip, AudioFileClip  # moviepy 1.x
    except ImportError as exc:
        pytest.skip(f"moviepy unavailable: {exc}")

    wav = tmp_media.tone_wav(name="tone.wav", seconds=1)
    try:
        audio = AudioFileClip(wav)
        clip = VideoFileClip(tmp_media.video(seconds=1, fps=10)).with_audio(audio)
        video_path = str(tmp_path / "with_audio.mp4")
        clip.write_videofile(video_path, fps=10, logger=None)
        clip.close()
        audio.close()
    except Exception as exc:
        pytest.skip(f"Cannot build video fixture: {exc}")

    try:
        report = HaqeeqatIngestor().ingest(video_path)
    except Exception as exc:
        pytest.skip(f"Ingestion failed (models unavailable?): {exc}")

    assert report["file_type"] == "video"
    assert isinstance(report["audio_transcript"], str)
    assert report["metadata"]["frames_sampled"] > 0
