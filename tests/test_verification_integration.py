import pytest

from verification.config import get_api_key
from verification.verdict_agent import VerdictAgent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not get_api_key(), reason="GROQ_API_KEY not set"),
]


def test_end_to_end_verdict_for_urdu_claim():
    result = VerdictAgent().run("کورونا ویکسین سے پانچ افراد کی موت ہو گئی")
    assert result.is_checkworthy is True
    assert result.verdict is not None
    assert 0.0 <= result.confidence <= 1.0
    assert result.reasoning_urdu.strip()
    assert result.evidence
