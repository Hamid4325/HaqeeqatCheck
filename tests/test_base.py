import pytest

from verification.base import (
    URDU_LABELS,
    EvidenceItem,
    SearchableClaim,
    Verdict,
    VerificationAgent,
    VerificationResult,
)


def test_verdict_enum_values():
    assert Verdict.SACHA.value == "sacha"
    assert Verdict.JHOOOTA.value == "jhoota"
    assert Verdict.MASHKOOK.value == "mashkook"


def test_urdu_labels_cover_all_verdicts():
    assert URDU_LABELS["sacha"] == "سچا"
    assert URDU_LABELS["jhoota"] == "جھوٹا"
    assert URDU_LABELS["mashkook"] == "مشکوک"
    assert set(URDU_LABELS) == {v.value for v in Verdict}


def test_verification_result_label_property():
    result = VerificationResult(verdict=Verdict.SACHA)
    assert result.verdict_label_urdu == "سچا"


def test_verification_result_label_empty_without_verdict():
    assert VerificationResult().verdict_label_urdu == ""


def test_evidence_item_fields():
    item = EvidenceItem(title="t", url="u", snippet="s", source_domain="afp.com")
    assert item.source_domain == "afp.com"


def test_searchable_claim_defaults_not_checkworthy():
    assert SearchableClaim().is_checkworthy is False


def test_verification_agent_is_abstract():
    with pytest.raises(TypeError):
        VerificationAgent()
