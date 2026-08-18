"""HaqeeqatIngestor: routes files to the right engine and builds the report."""

import os

from ingestion import utils
from ingestion.base import OCREngine, Transcriber, UnsupportedFormatError
from ingestion.cascade_ocr import CascadeOCREngine
from ingestion.whisper_transcriber import WhisperTranscriber

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv", ".3gp"}


class HaqeeqatIngestor:
    """Extracts all text (speech + on-screen) from an image, audio, or video.

    Engines are swappable: pass any object implementing Transcriber / OCREngine
    to the constructor. Defaults to WhisperTranscriber and CascadeOCREngine
    (UTRNet Urdu primary with PaddleOCR English fallback).
    """

    def __init__(
        self,
        transcriber: Transcriber | None = None,
        ocr_engine: OCREngine | None = None,
        frames_interval_sec: int = 5,
    ):
        self.transcriber = transcriber or WhisperTranscriber()
        self.ocr_engine = ocr_engine or CascadeOCREngine()
        self.frames_interval_sec = frames_interval_sec

    def ingest(self, path: str) -> dict:
        """Return the extraction report dict for the given media file."""
        if not os.path.isfile(path):
            raise UnsupportedFormatError(f"File not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGE_EXTS:
            return self._ingest_image(path)
        if ext in AUDIO_EXTS:
            return self._ingest_audio(path)
        if ext in VIDEO_EXTS:
            return self._ingest_video(path)
        raise UnsupportedFormatError(f"Unsupported file type: {ext or '(no extension)'}")

    def _base_metadata(self) -> dict:
        return {
            "whisper_model": getattr(self.transcriber, "model_size", None),
            "ocr_lang": getattr(self.ocr_engine, "lang", None),
            "ocr_engine_used": getattr(self.ocr_engine, "engine_used", None),
            "video_duration_sec": None,
            "frames_sampled": 0,
            "frames_interval_sec": self.frames_interval_sec,
            "warnings": [],
        }

    def _ingest_image(self, path: str) -> dict:
        ocr_text = self.ocr_engine.extract_text(path)
        garbled = utils.is_ocr_garbled(ocr_text)
        metadata = self._base_metadata()
        metadata["frames_sampled"] = 1
        metadata["ocr_garbled"] = garbled
        if garbled:
            metadata["warnings"] = metadata.get("warnings", []) + [
                "OCR produced garbled output; results may be unreliable"
            ]
        return {
            "file_type": "image",
            "audio_transcript": "",
            "ocr_text": ocr_text,
            "combined_text": utils.bidi_safe_combine("", ocr_text),
            "metadata": metadata,
        }

    def _ingest_audio(self, path: str) -> dict:
        transcript = self.transcriber.transcribe(path)
        return {
            "file_type": "audio",
            "audio_transcript": transcript,
            "ocr_text": "",
            "combined_text": utils.bidi_safe_combine(transcript, ""),
            "metadata": self._base_metadata(),
        }

    def _ingest_video(self, path: str) -> dict:
        warnings = []
        with utils.TempDirManager() as tmp_dir:
            transcript = ""
            audio_path = self._extract_audio(path, tmp_dir)
            if audio_path:
                transcript = self.transcriber.transcribe(audio_path)
            else:
                warnings.append("no audio track found")

            frames = utils.sample_video_frames(
                path, tmp_dir, interval_sec=self.frames_interval_sec
            )
            all_lines = []
            for frame_path in frames:
                all_lines.extend(self.ocr_engine.extract_text(frame_path).splitlines())
            ocr_text = "\n".join(utils.deduplicate_text(all_lines))
            if not frames:
                warnings.append("no extractable frames")

        metadata = self._base_metadata()
        metadata["video_duration_sec"] = utils.video_duration_sec(path)
        metadata["frames_sampled"] = len(frames)
        metadata["warnings"] = warnings
        return {
            "file_type": "video",
            "audio_transcript": transcript,
            "ocr_text": ocr_text,
            "combined_text": utils.bidi_safe_combine(transcript, ocr_text),
            "metadata": metadata,
        }

    def _extract_audio(self, video_path: str, output_dir: str) -> str | None:
        """Extract the audio track to a WAV file; returns its path or None.

        Import of moviepy is deferred so the module works without it installed
        (a video with no usable audio track simply yields a warning).
        """
        try:
            try:
                from moviepy.editor import VideoFileClip  # moviepy 1.x
            except ImportError:
                from moviepy import VideoFileClip  # moviepy 2.x
        except ImportError:
            return None

        output_path = os.path.join(output_dir, "extracted_audio.wav")
        try:
            clip = VideoFileClip(video_path)
            try:
                if clip.audio is None:
                    return None
                clip.audio.write_audiofile(output_path, logger=None)
            finally:
                clip.close()
            return output_path
        except Exception:
            return None
