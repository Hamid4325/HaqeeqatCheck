"""Abstract engine interfaces and module-level exceptions."""

from abc import ABC, abstractmethod


class Transcriber(ABC):
    """Converts an audio file into a text transcript."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Return the transcribed text of the audio file at audio_path."""


class OCREngine(ABC):
    """Extracts text from an image file."""

    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Return text found in the image at image_path."""


class UnsupportedFormatError(Exception):
    """Raised when a file cannot be ingested (unsupported or unreadable)."""
