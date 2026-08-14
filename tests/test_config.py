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
