import os
from types import SimpleNamespace

import app_gui
from verification.base import EvidenceItem, Verdict, VerificationResult


class _StopInterrupt(Exception):
    pass


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Sidebar:
    def __init__(self, outer):
        self._outer = outer

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self._outer._record(name, *args, **kwargs)

        return call


class FakeST:
    def __init__(self):
        self.log = []
        self.secrets = {"GROQ_API_KEY": ""}
        self.sidebar = _Sidebar(self)
        self.mode = 0
        self.text = "اردو متن"
        self.uploaded = None
        self.clicked = True

    def _record(self, name, *args, **kwargs):
        self.log.append((name, args, kwargs))

    def set_page_config(self, **kwargs):
        self.log.append(("set_page_config", (), kwargs))

    def title(self, *a, **k):
        self._record("title", *a, **k)

    def markdown(self, *a, **k):
        self._record("markdown", *a, **k)

    def code(self, *a, **k):
        self._record("code", *a, **k)

    def success(self, *a, **k):
        self._record("success", *a, **k)

    def error(self, *a, **k):
        self._record("error", *a, **k)

    def warning(self, *a, **k):
        self._record("warning", *a, **k)

    def info(self, *a, **k):
        self._record("info", *a, **k)

    def caption(self, *a, **k):
        self._record("caption", *a, **k)

    def header(self, *a, **k):
        self._record("header", *a, **k)

    def spinner(self, msg):
        self._record("spinner", msg)
        return _Ctx()

    def expander(self, label, **kwargs):
        self._record("expander", label)
        return _Ctx()

    def radio(self, label, options, **kwargs):
        self._record("radio", label, options)
        return options[self.mode]

    def text_area(self, label, **kwargs):
        self._record("text_area", label)
        return self.text

    def button(self, label):
        self._record("button", label)
        return self.clicked

    def file_uploader(self, label, **kwargs):
        self._record("file_uploader", label)
        return self.uploaded

    def stop(self):
        raise _StopInterrupt()


class FakeAgent:
    def __init__(self, result):
        self.result = result
        self.texts = []

    def run(self, text):
        self.texts.append(text)
        return self.result


class FakeIngestor:
    def __init__(self, combined_text="اردو متن"):
        self.combined_text = combined_text
        self.paths = []

    def ingest(self, path):
        self.paths.append(path)
        return {"combined_text": self.combined_text}


def _result(verdict, is_checkworthy=True, evidence=True):
    return VerificationResult(
        claim_urdu="دعویٰ",
        claim_english="claim",
        is_checkworthy=is_checkworthy,
        verdict=verdict,
        reasoning_urdu="وجوہات",
        reasoning_english="reasoning",
        confidence=0.8,
        evidence=(
            [
                EvidenceItem(
                    title="Soch",
                    url="https://sochfactcheck.com/a",
                    snippet="snip",
                    source_domain="sochfactcheck.com",
                )
            ]
            if evidence
            else []
        ),
    )


def _has(stub, name):
    return any(entry[0] == name for entry in stub.log)


def _markdown_texts(stub):
    return [
        args[0]
        for name, args, _ in stub.log
        if name == "markdown" and args and isinstance(args[0], str)
    ]


def test_sacha_uses_success_box():
    stub = FakeST()
    app_gui.main(st=stub, agent=FakeAgent(_result(Verdict.SACHA)))
    assert _has(stub, "success")
    assert not _has(stub, "error")
    assert not _has(stub, "warning")


def test_jhoota_uses_error_box():
    stub = FakeST()
    app_gui.main(st=stub, agent=FakeAgent(_result(Verdict.JHOOOTA)))
    assert _has(stub, "error")
    assert not _has(stub, "success")


def test_mashkook_uses_warning_box():
    stub = FakeST()
    app_gui.main(st=stub, agent=FakeAgent(_result(Verdict.MASHKOOK)))
    assert _has(stub, "warning")
    assert not _has(stub, "success")
    assert not _has(stub, "error")


def test_not_checkworthy_uses_info_box():
    stub = FakeST()
    app_gui.main(st=stub, agent=FakeAgent(_result(None, is_checkworthy=False)))
    assert _has(stub, "info")
    assert not _has(stub, "success")
    assert not _has(stub, "error")
    assert not _has(stub, "warning")


def test_text_mode_passes_text_to_agent():
    stub = FakeST()
    stub.text = "سندھ میں بارش"
    agent = FakeAgent(_result(Verdict.SACHA))
    app_gui.main(st=stub, agent=agent)
    assert agent.texts == ["سندھ میں بارش"]


def test_extracted_text_renders_wrapped_not_code_block():
    stub = FakeST()
    stub.text = "لمبی متن\nدوسری لائن"
    agent = FakeAgent(_result(Verdict.SACHA))
    app_gui.main(st=stub, agent=agent)
    assert not any(name == "code" for name, _, _ in stub.log)
    assert any(
        "extracted-text" in text and "لمبی متن" in text
        for text in _markdown_texts(stub)
    )


def test_media_mode_writes_temp_file_and_ingests(monkeypatch):
    stub = FakeST()
    stub.mode = 1
    stub.uploaded = SimpleNamespace(name="clip.mp4", getvalue=lambda: b"bytes")
    ingestor = FakeIngestor("نکالا ہوا متن")
    agent = FakeAgent(_result(Verdict.MASHKOOK))
    ensured = []
    monkeypatch.setattr(app_gui, "_ensure_models", lambda: ensured.append(True))
    app_gui.main(st=stub, ingestor=ingestor, agent=agent)
    assert ensured == [True]
    assert len(ingestor.paths) == 1
    assert ingestor.paths[0].endswith(".mp4")
    assert agent.texts == ["نکالا ہوا متن"]
    assert _has(stub, "spinner")


def test_evidence_rendered_in_sidebar():
    stub = FakeST()
    app_gui.main(st=stub, agent=FakeAgent(_result(Verdict.SACHA)))
    assert any("sochfactcheck.com" in text for text in _markdown_texts(stub))


def test_bilingual_claim_and_reasoning_rendered():
    stub = FakeST()
    app_gui.main(st=stub, agent=FakeAgent(_result(Verdict.SACHA)))
    assert any("reasoning" in text for text in _markdown_texts(stub))
    assert any("وجوہات" in text for text in _markdown_texts(stub))
    assert any("claim" in text for text in _markdown_texts(stub))
    assert any("دعویٰ" in text for text in _markdown_texts(stub))


def test_secrets_fallback_populates_env(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    stub = FakeST()
    stub.secrets = {"GROQ_API_KEY": "sk-test"}
    app_gui.main(st=stub, agent=FakeAgent(_result(Verdict.SACHA)))
    assert os.environ["GROQ_API_KEY"] == "sk-test"
