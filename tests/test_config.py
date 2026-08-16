import pytest

from verification import config


def test_get_api_key_reads_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert config.get_api_key() == "gsk_test"


def test_get_api_key_defaults_to_empty(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert config.get_api_key() == ""


def test_default_model_id():
    assert config.MODEL_ID == "llama-3.3-70b-versatile"


def test_source_priority_is_strictly_ordered():
    assert config.SOURCE_PRIORITY == ["sochfactcheck.com", "afp.com", "dawn.com"]


def test_jhoota_min_confidence_threshold():
    assert config.JHOOOTA_MIN_CONFIDENCE == 0.5


def test_debug_enabled(monkeypatch):
    monkeypatch.delenv("HAQEEQAT_DEBUG", raising=False)
    assert config.debug_enabled() is False
    monkeypatch.setenv("HAQEEQAT_DEBUG", "1")
    assert config.debug_enabled() is True
    monkeypatch.setenv("HAQEEQAT_DEBUG", "yes")
    assert config.debug_enabled() is True
    monkeypatch.setenv("HAQEEQAT_DEBUG", "0")
    assert config.debug_enabled() is False
