"""Whisper-based speech-to-text engine."""

import os

from ingestion.base import Transcriber

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HF_MODELS_DIR = "/home/user/app/models"
_LOCAL_MODELS_DIR = os.path.join(_REPO_ROOT, "models")


def _whisper_download_root() -> str:
    """Return the best directory for caching Whisper model weights."""
    for candidate in (_HF_MODELS_DIR, _LOCAL_MODELS_DIR):
        whisper_dir = os.path.join(candidate, "whisper")
        if os.path.isdir(whisper_dir) and os.listdir(whisper_dir):
            return whisper_dir
    # Prefer HF path on Hugging Face Spaces, else local
    if os.path.isdir(_HF_MODELS_DIR) or os.environ.get("SPACE_ID"):
        return os.path.join(_HF_MODELS_DIR, "whisper")
    return os.path.join(_LOCAL_MODELS_DIR, "whisper")


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

            self._model = whisper.load_model(
                self.model_size, download_root=_whisper_download_root()
            )
        return self._model

    def transcribe(self, audio_path: str) -> str:
        model = self._get_model()
        result = model.transcribe(audio_path)
        text = result.get("text") or ""
        return text.strip()
