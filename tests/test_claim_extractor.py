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
    assert fake.calls[0]["temperature"] == 0


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


def test_prompt_requires_attributed_quote_check():
    assert "attributed quote" in CLAIM_EXTRACTION_SYSTEM_PROMPT
    assert "did X actually say this" in CLAIM_EXTRACTION_SYSTEM_PROMPT


def test_forces_checkworthy_for_attributed_quote():
    fake = FakeGroqClient([VALID.replace("true", "false")])
    text = "میرا بس چلے\nمیرا بس چلے تو میں پیتہ نہیں\nآپ کو کیا کیا دے دوں! مریم نوز"
    result = ClaimExtractor(groq_client=fake).extract(text)
    assert result.is_checkworthy is True


def test_attribution_hint_in_user_message():
    fake = FakeGroqClient([VALID])
    ClaimExtractor(groq_client=fake).extract("میرا بس چلے — مریم نواز")
    user_content = fake.calls[0]["messages"][-1]["content"]
    assert "HINT:" in user_content
    assert "مریم نواز" in user_content


def test_parse_failure_keeps_attributed_quote_checkworthy():
    fake = FakeGroqClient(["bad", "also bad", "still bad"])
    text = "میرا بس چلے\nمیرا بس چلے تو میں پیتہ نہیں\nآپ کو کیا کیا دے دوں! مریم نوز"
    result = ClaimExtractor(groq_client=fake).extract(text)
    assert len(fake.calls) == 3
    assert result.is_checkworthy is True


def test_urdu_role_word_ke_mutabiq_rejected():
    text = "وزیرِ اعلیٰ پنجاب کے مطابق، بارشوں سے تین افراد ہلاک ہوئے"
    assert detect_attribution(text) is None


def test_urdu_spokesman_keha_not_attribution():
    text = "ترجمان نے کہا کہ پاکستان نے ترکی کے ساتھ دفاعی معاہدے میں توسیع کا اشارہ دیا"
    assert detect_attribution(text) is None


def test_garbled_spokesman_keha_not_attribution():
    text = "ترجوان نے کہا کہ پاکستان نے ترکی کے ساتھ دفاعی معاہدے میں توسیع کا اشارہ دیا"
    assert detect_attribution(text) is None


def test_english_role_word_not_attribution():
    assert detect_attribution("The spokesman said the deal was extended.") is None
    assert detect_attribution("Government said the deal was signed.") is None


def test_role_word_signature_rejected():
    assert detect_attribution("یہ ہمارا فیصلہ ہے۔ — حکومت") is None


def test_rejected_role_match_does_not_block_real_name():
    text = "وزیرِ اعلیٰ پنجاب نے کہا کہ مریم نواز نے کہا کہ بارشیں ہوئیں"
    assert detect_attribution(text) == "مریم نواز"


def test_role_word_news_has_no_attribution_hint():
    fake = FakeGroqClient([VALID])
    ClaimExtractor(groq_client=fake).extract(
        "ترجمان نے کہا کہ پاکستان نے ترکی کے ساتھ دفاعی معاہدے میں توسیع کا اشارہ دیا"
    )
    user_content = fake.calls[0]["messages"][-1]["content"]
    assert "HINT:" not in user_content


def test_prompt_rejects_role_only_attribution():
    assert "PROHIBITED" in CLAIM_EXTRACTION_SYSTEM_PROMPT
    assert "never frame a claim as an attributed quote" in CLAIM_EXTRACTION_SYSTEM_PROMPT


ROLE_QUOTE_RESPONSE = (
    '{"is_checkworthy": true, "urdu_claim": "کیا ترجمان نے یہ کہا کہ پاکستان '
    'نے دفاعی معاہدے میں توسیع کی؟", "english_claim": "Did the spokesman say '
    'that Pakistan expanded the defense agreement?"}'
)

REAL_QUOTE_RESPONSE = (
    '{"is_checkworthy": true, "urdu_claim": "کیا مریم نواز نے یہ کہا: میرا بس '
    'چلے؟", "english_claim": "Did Maryam Nawaz say: \\"Mera bus chaley\\"?"}'
)


def test_reframes_role_attribution_question_to_factual_claim():
    fake = FakeGroqClient([ROLE_QUOTE_RESPONSE])
    result = ClaimExtractor(groq_client=fake).extract(
        "ترجمان نے کہا کہ پاکستان نے دفاعی معاہدے میں توسیع کی"
    )
    assert result.is_checkworthy is True
    assert result.urdu_claim == "پاکستان نے دفاعی معاہدے میں توسیع کی"
    assert result.english_claim == "Pakistan expanded the defense agreement"


def test_keeps_real_name_attribution_question():
    fake = FakeGroqClient([REAL_QUOTE_RESPONSE])
    result = ClaimExtractor(groq_client=fake).extract("میرا بس چلے — مریم نواز")
    assert result.urdu_claim == "کیا مریم نواز نے یہ کہا: میرا بس چلے؟"
    assert result.english_claim.startswith("Did Maryam Nawaz say")


def test_keeps_plain_news_claim_unreframed():
    fake = FakeGroqClient([VALID])
    result = ClaimExtractor(groq_client=fake).extract("سندھ میں بارش")
    assert result.urdu_claim == "سندھ میں بارش سے تین افراد ہلاک ہو گئے"
    assert result.english_claim == "Three people died in rains in Sindh, Pakistan."


def test_detects_urdu_qa_qol_attribution():
    text = "مریم نواز کا قول ہے کہ تعلیم سب سے اہم ہے"
    assert detect_attribution(text) == "مریم نواز"


def test_signature_with_rtl_format_wrappers():
    text = (
        "\u2067میرابس چلے\n"
        "میرا بس چلے تو میں پیتہ نہیں\n"
        "آپ کو کیا کیا دے دوں ! مریم نوز\u2069"
    )
    assert detect_attribution(text) == "مریم نوز"
