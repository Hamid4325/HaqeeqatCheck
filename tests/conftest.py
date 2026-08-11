import math
import struct
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest


class _MediaFactory:
    def __init__(self, base: Path):
        self.base = base

    def image(self, name="img.png", size=(64, 64)):
        path = self.base / name
        img = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
        cv2.putText(img, "Urdu Test", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imwrite(str(path), img)
        return str(path)

    def tone_wav(self, name="tone.wav", seconds=1, freq=440, sample_rate=16000):
        path = self.base / name
        with wave.open(str(path), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            frames = bytearray()
            for i in range(sample_rate * seconds):
                sample = int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
                frames += struct.pack("<h", sample)
            w.writeframes(bytes(frames))
        return str(path)

    def video(self, name="clip.avi", fps=15, seconds=2, size=(64, 64)):
        path = self.base / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        for i in range(fps * seconds):
            frame = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
            cv2.putText(frame, str(i), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            writer.write(frame)
        writer.release()
        return str(path)


@pytest.fixture
def tmp_media(tmp_path):
    return _MediaFactory(tmp_path)
