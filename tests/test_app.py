from verification.app import main
from verification.base import EvidenceItem, Verdict, VerificationResult


class FakeIngestor:
    def __init__(self, combined_text):
        self.combined_text = combined_text

    def ingest(self, path):
        return {"combined_text": self.combined_text}


def test_usage_error_when_no_arg(capsys):
    assert main([], ingestor=None, agent=None) == 2
    assert "Usage" in capsys.readouterr().err


def test_exits_when_no_text(capsys):
    code = main(["prog", "file.png"], ingestor=FakeIngestor("   "))
    assert code == 1


def test_reports_not_checkworthy(capsys):
    agent = _StubAgent(VerificationResult(is_checkworthy=False))
    code = main(["prog", "f.png"], ingestor=FakeIngestor("x"), agent=agent)
    assert code == 0
    assert "کوئی قابلِ تصدیق دعویٰ نہیں" in capsys.readouterr().out


def test_prints_verdict_and_sources(capsys):
    result = VerificationResult(
        claim_urdu="دعویٰ",
        claim_english="claim",
        is_checkworthy=True,
        verdict=Verdict.SACHA,
        reasoning_urdu="وجوہات",
        confidence=0.9,
        evidence=[EvidenceItem("t", "https://sochfactcheck.com/a", "s", "sochfactcheck.com")],
    )
    code = main(["prog", "f.png"], ingestor=FakeIngestor("x"), agent=_StubAgent(result))
    out = capsys.readouterr().out
    assert code == 0
    assert "دعویٰ" in out
    assert "سچا" in out
    assert "sochfactcheck.com" in out


def test_debug_flag_enables_trace(monkeypatch, capsys):
    monkeypatch.delenv("HAQEEQAT_DEBUG", raising=False)
    result = VerificationResult(
        claim_urdu="دعویٰ",
        claim_english="claim",
        is_checkworthy=True,
        verdict=Verdict.SACHA,
        reasoning_urdu="وجوہات",
        confidence=0.9,
        evidence=[EvidenceItem("t", "https://sochfactcheck.com/a", "s", "sochfactcheck.com")],
    )
    code = main(["prog", "f.png", "--debug"], ingestor=FakeIngestor("x"), agent=_StubAgent(result))
    assert code == 0
    out = capsys.readouterr().out
    assert "DEBUG" in out
    assert "sacha" in out
    assert "sochfactcheck.com" in out


def test_debug_flag_works_before_path(monkeypatch):
    result = VerificationResult(
        claim_urdu="دعویٰ",
        claim_english="claim",
        is_checkworthy=True,
        verdict=Verdict.SACHA,
        reasoning_urdu="وجوہات",
        confidence=0.9,
    )
    code = main(
        ["prog", "--debug", "f.png"], ingestor=FakeIngestor("x"), agent=_StubAgent(result)
    )
    assert code == 0


class _StubAgent:
    def __init__(self, result):
        self.result = result

    def run(self, text):
        return self.result
