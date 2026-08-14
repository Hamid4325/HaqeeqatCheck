import pytest

from tests.fakes import FakeGroqClient
from verification.claim_extractor import CLAIM_EXTRACTION_SYSTEM_PROMPT, ClaimExtractor

VALID = (
    '{"is_checkworthy": true, "urdu_claim": "سندھ میں بارش سے تین افراد ہلاک ہو گئے",'
    ' "english_claim": "Three people died in rains in Sindh, Pakistan."}'
)


def test_extracts_checkworthy_claim():
    fake = FakeGroqClient([VALID])
    result = ClaimExtractor(groq_client=fake).extract("barish se teen afrad halak")
    assert result.is_checkworthy is True
    assert result.urdu_claim == "سندھ میں بارش سے تین افراد ہلاک ہو گئے"
    assert result.english_claim.startswith("Three people")
    assert fake.calls[0]["model"] == "llama-3.3-70b-versatile"
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


def test_prompt_requires_roman_urdu_transliteration():
    assert "Roman Urdu" in CLAIM_EXTRACTION_SYSTEM_PROMPT
    assert "اردو رسم الخط" in CLAIM_EXTRACTION_SYSTEM_PROMPT


def test_input_is_wrapped_in_text_delimiters():
    fake = FakeGroqClient([VALID])
    ClaimExtractor(groq_client=fake).extract("some text")
    user_content = fake.calls[0]["messages"][-1]["content"]
    assert user_content == "<text>\nsome text\n</text>"


def test_ignores_non_checkworthy_text():
    fake = FakeGroqClient([VALID.replace("true", "false")])
    result = ClaimExtractor(groq_client=fake).extract("السلام علیکم")
    assert result.is_checkworthy is False


def test_retries_on_invalid_json_then_succeeds():
    fake = FakeGroqClient(["not json at all", VALID])
    result = ClaimExtractor(groq_client=fake).extract("some text")
    assert result.is_checkworthy is True
    assert len(fake.calls) == 2


def test_not_checkworthy_after_retries_exhausted():
    fake = FakeGroqClient(["bad", "also bad", "still bad"])
    result = ClaimExtractor(groq_client=fake).extract("some text")
    assert result.is_checkworthy is False
    assert len(fake.calls) == 3


def test_missing_api_key_raises_friendly_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        ClaimExtractor().extract("some text")
