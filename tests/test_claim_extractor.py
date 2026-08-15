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


from verification.claim_extractor import detect_attribution


def test_detects_urdu_explicit_attribution():
    text = "مریم نواز نے کہا کہ سندھ میں بارش سے تین افراد ہلاک ہو گئے"
    assert detect_attribution(text) == "مریم نواز"


def test_detects_english_explicit_attribution():
    text = "Maryam Nawaz said that she will visit Lahore."
    assert detect_attribution(text) == "Maryam Nawaz"


def test_detects_dash_signature():
    assert detect_attribution("میرا بس چلے۔ — مریم نواز") == "مریم نواز"


def test_detects_exclamation_signature_with_ocr_noise():
    text = "میرا بس چلے\nمیرا بس چلے تو میں پیتے نہیں\nآپ کو کیا کیا دے دوں! مریم نوز"
    assert detect_attribution(text) == "مریم نوز"


def test_detects_colon_quote_prefix():
    assert detect_attribution('مریم نواز: "یہ سب جھوٹ ہے"') == "مریم نواز"


def test_returns_none_for_plain_news_text():
    assert detect_attribution("سندھ میں بارش سے تین افراد ہلاک ہو گئے") is None


def test_returns_none_for_greeting():
    assert detect_attribution("السلام علیکم") is None
