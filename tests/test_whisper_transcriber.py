import sys
import types

from ingestion.whisper_transcriber import WhisperTranscriber


def test_transcribe_strips_whitespace(monkeypatch, tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"wav")

    fake_model = types.SimpleNamespace(
        transcribe=lambda path: {"text": "  hello duniya  "}
    )
    fake_whisper = types.ModuleType("whisper")
    fake_whisper.load_model = lambda size: fake_model
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    transcriber = WhisperTranscriber(model_size="base")
    assert transcriber.transcribe(str(audio)) == "hello duniya"
    assert transcriber._model is fake_model


def test_model_loaded_once(monkeypatch, tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"wav")
    load_count = 0

    def load_model(size):
        nonlocal load_count
        load_count += 1
        return types.SimpleNamespace(transcribe=lambda p: {"text": "x"})

    fake_whisper = types.ModuleType("whisper")
    fake_whisper.load_model = load_model
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    transcriber = WhisperTranscriber()
    transcriber.transcribe(str(audio))
    transcriber.transcribe(str(audio))
    assert load_count == 1
