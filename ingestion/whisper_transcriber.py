"""Whisper-based speech-to-text engine."""

from ingestion.base import Transcriber


class WhisperTranscriber(Transcriber):
    """Transcribes audio using openai-whisper.

    The model is loaded lazily on first transcribe() call so importing this
    module stays fast and model downloads don't happen at import time.

    Install note:
        pip install openai-whisper
    ffmpeg must also be on your PATH (already present on this machine).

    The 'base' multilingual model auto-detects language, so it handles Urdu
    and English. Raise the model size (e.g. 'small') for better accuracy at
    the cost of speed and memory.
    """

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            import whisper

            self._model = whisper.load_model(self.model_size)
        return self._model

    def transcribe(self, audio_path: str) -> str:
        model = self._get_model()
        result = model.transcribe(audio_path)
        text = result.get("text") or ""
        return text.strip()
