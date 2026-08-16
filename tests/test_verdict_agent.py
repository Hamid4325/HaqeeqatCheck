from tests.fakes import FakeGroqClient
from verification.base import Verdict
from verification.claim_extractor import ClaimExtractor
from verification.evidence_retriever import EvidenceRetriever
from verification.verdict_agent import VerdictAgent

CLAIM = (
    '{"is_checkworthy": true, "urdu_claim": "سندھ میں بارش سے تین افراد ہلاک ہو گئے",'
    ' "english_claim": "Three people died in rains in Sindh, Pakistan."}'
)
VERDICT = (
    '{"verdict": "sacha", "reasoning_urdu": "ثبوت اس دعوے کی حمایت کرتے ہیں۔",'
    ' "reasoning_english": "The evidence supports this claim.",'
    ' "confidence": 0.9}'
)
JHOOOTA_LOW = VERDICT.replace('"sacha"', '"jhoota"').replace("0.9", "0.3")
JHOOOTA_HIGH = VERDICT.replace('"sacha"', '"jhoota"')
RESULT = {"title": "Soch", "href": "https://sochfactcheck.com/a", "body": "x"}


class _Search:
    def __init__(self, batches):
        self.batches = batches

    def __call__(self, query, region="pk-en", max_results=5):
        return self.batches.pop(0)


def _agent(groq_contents, search_batches=None):
    fake = FakeGroqClient(groq_contents)
    agent = VerdictAgent(
        claim_extractor=ClaimExtractor(groq_client=fake),
        groq_client=fake,
    )
    if search_batches is not None:
        agent.evidence_retriever = EvidenceRetriever(search_fn=_Search(search_batches))
    return agent, fake


def test_short_circuits_when_not_checkworthy():
    fake = FakeGroqClient([CLAIM.replace("true", "false")])
    agent = VerdictAgent(claim_extractor=ClaimExtractor(groq_client=fake))
    result = agent.run("السلام علیکم")
    assert result.is_checkworthy is False
    assert result.verdict is None
    assert len(fake.calls) == 1


def test_no_evidence_returns_mashkook_fallback():
    agent, fake = _agent([CLAIM], [[]])
    result = agent.run("some text")
    assert result.verdict is Verdict.MASHKOOK
    assert result.verdict_label_urdu == "مشکوک"
    assert result.confidence == 0.3
    assert result.reasoning_urdu
    assert len(fake.calls) == 1


def test_parses_valid_verdict():
    agent, fake = _agent([CLAIM, VERDICT], [[RESULT]])
    result = agent.run("some text")
    assert result.is_checkworthy is True
    assert result.verdict is Verdict.SACHA
    assert result.verdict_label_urdu == "سچا"
    assert result.confidence == 0.9
    assert result.reasoning_urdu == "ثبوت اس دعوے کی حمایت کرتے ہیں۔"
    assert result.reasoning_english == "The evidence supports this claim."
    assert result.evidence[0].source_domain == "sochfactcheck.com"
    assert fake.calls[1]["temperature"] == 0


def test_bad_verdict_value_defaults_to_mashkook():
    agent, _ = _agent(
        [CLAIM, VERDICT.replace('"sacha"', '"jhoota1"')], [[RESULT]]
    )
    assert agent.run("t").verdict is Verdict.MASHKOOK


def test_jhoota_below_confidence_threshold_downgrades_to_mashkook():
    agent, _ = _agent([CLAIM, JHOOOTA_LOW], [[RESULT]])
    result = agent.run("t")
    assert result.verdict is Verdict.MASHKOOK


def test_jhoota_high_confidence_stays_jhoota():
    agent, _ = _agent([CLAIM, JHOOOTA_HIGH], [[RESULT]])
    assert agent.run("t").verdict is Verdict.JHOOOTA


def test_prompt_guard_against_false_from_absence_of_evidence():
    from verification.verdict_agent import VERDICT_SYSTEM_PROMPT

    prompt = VERDICT_SYSTEM_PROMPT.lower()
    assert "contradict" in prompt
    assert "never" in prompt
    assert "mashkook" in prompt


def test_confidence_is_clamped():
    agent, _ = _agent([CLAIM, VERDICT.replace("0.9", "7.5")], [[RESULT]])
    assert agent.run("t").confidence == 1.0
    agent, _ = _agent([CLAIM, VERDICT.replace("0.9", "-3")], [[RESULT]])
    assert agent.run("t").confidence == 0.0


def test_retries_on_invalid_verdict_json():
    agent, fake = _agent([CLAIM, "garbage", VERDICT], [[RESULT]])
    result = agent.run("t")
    assert result.verdict is Verdict.SACHA
    assert len(fake.calls) == 3
