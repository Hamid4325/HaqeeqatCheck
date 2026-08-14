# Verification & Reasoning Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `verification/` package that turns `combined_text` into an Urdu verdict (سچا / جھوٹا / مشکوک) using Groq Llama-3.3-70B and duckduckgo-search snippets.

**Architecture:** Three chained agents — `ClaimExtractor` (Groq → one SearchableClaim), `EvidenceRetriever` (dual-language DDG search, deduped by URL, strictly prioritized), `VerdictAgent` (Groq → VerificationResult) — composed by a `verification/app.py` CLI. All external calls are injected (constructor args), so unit tests are offline.

**Tech Stack:** Python 3.12, `groq` (llama-3.3-70b-versatile, JSON mode), `duckduckgo-search`, `python-dotenv`, pytest.

## Global Constraints

- Work directly on `main` (no worktree); no secrets committed — `.env` is gitignored.
- Only free/open-source dependencies; model imports stay lazy (imported inside functions).
- The existing 65-test fast suite must keep passing; new integration tests must auto-skip.
- Roman Urdu input must be converted to proper Urdu script (اردو رسم الخط) in `urdu_claim`.
- Evidence domains strictly prioritized: `sochfactcheck.com` > `afp.com` > `dawn.com` > all others.
- No fabricated verdicts: empty evidence → `mashkook` fallback, never invent sources.

---

### Task 1: Scaffold package, config, dependencies

**Files:**
- Create: `verification/__init__.py`
- Create: `verification/config.py`
- Test: `tests/test_config.py`
- Modify: `requirements.txt`
- Modify: `pytest.ini`

**Interfaces:**
- Consumes: nothing.
- Produces: `verification/config.py` constants `MODEL_ID`, `MAX_RETRIES`, `MAX_QUERY_RESULTS`, `SOURCE_PRIORITY`, `NO_EVIDENCE_CONFIDENCE`, and function `get_api_key() -> str`. `verification/` package importable.

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verification'`

- [ ] **Step 3: Write minimal implementation**

`verification/__init__.py` (exports are added in Task 2):

```python
"""Verification & Reasoning Engine module (Haqeeqat Check, Module 2)."""
```

`verification/config.py`:

```python
import os
from pathlib import Path

from dotenv import load_dotenv

MODEL_ID = "llama-3.3-70b-versatile"
MAX_RETRIES = 3
MAX_QUERY_RESULTS = 5
SOURCE_PRIORITY = ["sochfactcheck.com", "afp.com", "dawn.com"]
NO_EVIDENCE_CONFIDENCE = 0.3
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0


def _find_env_file() -> Path | None:
    for directory in (Path.cwd(), Path(__file__).resolve().parent.parent):
        env_path = directory / ".env"
        if env_path.exists():
            return env_path
    return None


_env_path = _find_env_file()
if _env_path:
    load_dotenv(_env_path)


def get_api_key() -> str:
    return os.environ.get("GROQ_API_KEY", "")
```

`requirements.txt` — append these lines:

```
groq
duckduckgo-search
python-dotenv
```

`pytest.ini` — change the markers block to:

```
[pytest]
markers =
    slow: long-running tests that download models (auto-skip on failure)
    integration: tests that hit the network or require GROQ_API_KEY (auto-skip)
```

Install deps:

Run: `python -m pip install groq duckduckgo-search python-dotenv`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add verification/__init__.py verification/config.py tests/test_config.py requirements.txt pytest.ini
git commit -m "feat(verification): scaffold package and config"
```

---

### Task 2: base.py — types, enum, ABC

**Files:**
- Create: `verification/base.py`
- Test: `tests/test_base.py`
- Modify: `verification/__init__.py`

**Interfaces:**
- Consumes: Task 1 package.
- Produces: `Verdict` (str-enum), `URDU_LABELS: dict[str, str]`, `EvidenceItem`, `SearchableClaim`, `VerificationResult`, `VerificationAgent(ABC)` with `run(self, text: str) -> VerificationResult`. `VerificationResult.verdict_label_urdu` property.

- [ ] **Step 1: Write the failing test**

`tests/test_base.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verification.base'`

- [ ] **Step 3: Write minimal implementation**

`verification/base.py`:

```python
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class Verdict(str, enum.Enum):
    SACHA = "sacha"
    JHOOOTA = "jhoota"
    MASHKOOK = "mashkook"


URDU_LABELS = {
    "sacha": "سچا",
    "jhoota": "جھوٹا",
    "mashkook": "مشکوک",
}


@dataclass
class EvidenceItem:
    title: str
    url: str
    snippet: str
    source_domain: str


@dataclass
class SearchableClaim:
    is_checkworthy: bool = False
    urdu_claim: str = ""
    english_claim: str = ""


@dataclass
class VerificationResult:
    claim_urdu: str = ""
    claim_english: str = ""
    is_checkworthy: bool = False
    verdict: Verdict | None = None
    reasoning_urdu: str = ""
    confidence: float = 0.0
    evidence: list[EvidenceItem] = field(default_factory=list)

    @property
    def verdict_label_urdu(self) -> str:
        if self.verdict is None:
            return ""
        return URDU_LABELS[self.verdict.value]


class VerificationAgent(ABC):
    @abstractmethod
    def run(self, text: str) -> VerificationResult:
        ...
```

Update `verification/__init__.py`:

```python
from .base import (
    URDU_LABELS,
    EvidenceItem,
    SearchableClaim,
    Verdict,
    VerificationAgent,
    VerificationResult,
)

__all__ = [
    "URDU_LABELS",
    "EvidenceItem",
    "SearchableClaim",
    "Verdict",
    "VerificationAgent",
    "VerificationResult",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_base.py -v`
Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add verification/base.py verification/__init__.py tests/test_base.py
git commit -m "feat(verification): add verdict enum, result types, agent ABC"
```

---

### Task 3: claim_extractor.py — extraction with Roman Urdu conversion

**Files:**
- Create: `verification/claim_extractor.py`
- Create: `tests/fakes.py`
- Test: `tests/test_claim_extractor.py`

**Interfaces:**
- Consumes: Task 1 `config.get_api_key()`, `MODEL_ID`, `MAX_RETRIES`; Task 2 `SearchableClaim`.
- Produces: `CLAIM_EXTRACTION_SYSTEM_PROMPT: str` and `class ClaimExtractor` with `__init__(self, groq_client=None, model=MODEL_ID, max_retries=MAX_RETRIES)` and `extract(self, text: str) -> SearchableClaim`. Raises `RuntimeError` when `GROQ_API_KEY` is missing and no client injected.

- [ ] **Step 1: Write the failing test**

`tests/fakes.py`:

```python
from types import SimpleNamespace


class FakeGroqClient:
    """Stand-in for groq.Groq; returns canned chat-completion contents."""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = []

    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._contents:
            content = self._contents.pop(0)
        else:
            content = (
                '{"is_checkworthy": false, "urdu_claim": "", "english_claim": ""}'
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
```

`tests/test_claim_extractor.py`:

```python
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
    assert result.urdu_claim == ""


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_claim_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verification.claim_extractor'`

- [ ] **Step 3: Write minimal implementation**

`verification/claim_extractor.py`:

```python
import json
import re

from .base import SearchableClaim
from .config import MAX_RETRIES, MODEL_ID, get_api_key

CLAIM_EXTRACTION_SYSTEM_PROMPT = """\
You are the claim-extraction step of an Urdu fact-checking pipeline.
You receive raw, noisy text: OCR from an image, a speech-to-text transcript, or
both. The text may be written in Urdu script (اردو رسم الخط) OR in Roman Urdu
(Urdu written with English/Latin letters, e.g. "sarkar ne naya qanoon pass kiya").
Decide whether the text contains ONE verifiable factual claim, and if so extract it.

RULES:
1. A claim is check-worthy ONLY if it asserts a verifiable fact about the real
   world: politics, economy, health, religion, sports, crime, viral rumors, etc.
2. IGNORE as not check-worthy: greetings, pleasantries, personal opinions,
   poetry, personal stories or experiences, pure questions, jokes, and any
   content with no factual assertion.
3. If several claims exist, extract only the SINGLE most viral or important one.
4. "urdu_claim": the claim as ONE concise sentence, ALWAYS written in proper
   Urdu script (اردو رسم الخط). If the original is in Roman Urdu, transliterate
   it into proper Urdu script. Preserve the original wording where possible;
   never add your own interpretation.
5. "english_claim": a faithful English translation of urdu_claim, written for
   international web search (mention "Pakistan" if the claim concerns it).
6. "is_checkworthy": true ONLY if a verifiable claim passes rules 1-5.

Respond ONLY with a JSON object, exactly this shape (no prose, no markdown):
{"is_checkworthy": true, "urdu_claim": "...", "english_claim": "..."}"""


class ClaimExtractor:
    def __init__(self, groq_client=None, model=MODEL_ID, max_retries=MAX_RETRIES):
        self._client = groq_client
        self.model = model
        self.max_retries = max_retries

    def extract(self, text: str) -> SearchableClaim:
        messages = [
            {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"<text>\n{text}\n</text>"},
        ]
        parsed = self._chat(messages)
        if parsed is None:
            return SearchableClaim(is_checkworthy=False)
        return SearchableClaim(
            is_checkworthy=parsed["is_checkworthy"],
            urdu_claim=parsed["urdu_claim"],
            english_claim=parsed["english_claim"],
        )

    def _chat(self, messages):
        client = self._get_client()
        last_content = ""
        for _ in range(self.max_retries):
            call_messages = list(messages)
            if last_content:
                call_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Previous output was invalid: {last_content}\n"
                            "Return valid JSON only, in exactly the specified shape."
                        ),
                    }
                )
            response = client.chat.completions.create(
                model=self.model,
                messages=call_messages,
                temperature=0.2,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            last_content = response.choices[0].message.content or ""
            parsed = self._parse(last_content)
            if parsed is not None:
                return parsed
        return None

    def _parse(self, content: str):
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{.*\}", content or "", re.DOTALL)
            if match is None:
                return None
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        if not isinstance(data, dict):
            return None
        is_checkworthy = data.get("is_checkworthy")
        urdu_claim = data.get("urdu_claim")
        english_claim = data.get("english_claim")
        if not isinstance(is_checkworthy, bool):
            return None
        if not isinstance(urdu_claim, str) or not urdu_claim.strip():
            return None
        if not isinstance(english_claim, str) or not english_claim.strip():
            return None
        return {
            "is_checkworthy": is_checkworthy,
            "urdu_claim": urdu_claim.strip(),
            "english_claim": english_claim.strip(),
        }

    def _get_client(self):
        if self._client is None:
            from groq import Groq

            api_key = get_api_key()
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Create a .env file at the project "
                    "root with GROQ_API_KEY=<your key>."
                )
            self._client = Groq(api_key=api_key)
        return self._client
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_claim_extractor.py -v`
Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add verification/claim_extractor.py tests/fakes.py tests/test_claim_extractor.py
git commit -m "feat(verification): add claim extractor with Roman Urdu handling"
```

---

### Task 4: evidence_retriever.py — dual-language search, dedupe, strict priority

**Files:**
- Create: `verification/evidence_retriever.py`
- Test: `tests/test_evidence_retriever.py`

**Interfaces:**
- Consumes: Task 2 `EvidenceItem`; Task 1 `MAX_QUERY_RESULTS`, `SOURCE_PRIORITY`.
- Produces: `class EvidenceRetriever` with `__init__(self, search_fn=None, max_results=MAX_QUERY_RESULTS)` and `retrieve(self, urdu_claim: str, english_claim: str) -> list[EvidenceItem]`. `search_fn(query, region, max_results)` mirrors `DDGS().text`; results are dicts with `title`, `href`, `body`.

- [ ] **Step 1: Write the failing test**

`tests/test_evidence_retriever.py`:

```python
from verification.evidence_retriever import EvidenceRetriever

A = {"title": "Soch A", "href": "https://sochfactcheck.com/a", "body": "snippet a"}
AFP = {"title": "AFP B", "href": "https://factcheck.afp.com/doc/b", "body": "snippet b"}
DAWN = {"title": "Dawn C", "href": "https://www.dawn.com/c", "body": "snippet c"}
OTHER = {"title": "X D", "href": "https://example.com/d", "body": "snippet d"}


class RecordingSearch:
    def __init__(self, *batches):
        self.batches = list(batches)
        self.queries = []

    def __call__(self, query, region="pk-en", max_results=5):
        self.queries.append(query)
        if not self.batches:
            return []
        return self.batches.pop(0)


def test_runs_two_language_queries():
    search = RecordingSearch([], [])
    EvidenceRetriever(search_fn=search).retrieve("اردو دعویٰ", "English claim")
    assert search.queries == ["English claim fact check", "اردو دعویٰ"]


def test_dedupes_by_url_later_replaces_earlier():
    other = dict(A, href="https://sochfactcheck.com/a", body="earlier snippet")
    later = dict(A, href="https://sochfactcheck.com/a", body="later snippet")
    search = RecordingSearch([other], [later])
    results = EvidenceRetriever(search_fn=search).retrieve("u", "e")
    assert len(results) == 1
    assert results[0].snippet == "later snippet"


def test_strictly_prioritizes_soch_afp_dawn():
    search = RecordingSearch([DAWN, OTHER, AFP, A], [])
    results = EvidenceRetriever(search_fn=search).retrieve("u", "e")
    domains = [r.source_domain for r in results]
    assert domains.index("sochfactcheck.com") < domains.index("factcheck.afp.com")
    assert domains.index("factcheck.afp.com") < domains.index("www.dawn.com")
    assert domains.index("www.dawn.com") < domains.index("example.com")


def test_missing_href_results_are_dropped():
    search = RecordingSearch([{"title": "no url", "body": "x"}, A], [])
    results = EvidenceRetriever(search_fn=search).retrieve("u", "e")
    assert len(results) == 1


def test_rate_limit_degrades_to_empty():
    def boom(query, region="pk-en", max_results=5):
        raise RuntimeError("ratelimit")

    assert EvidenceRetriever(search_fn=boom).retrieve("u", "e") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evidence_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verification.evidence_retriever'`

- [ ] **Step 3: Write minimal implementation**

`verification/evidence_retriever.py`:

```python
from urllib.parse import urlparse

from .base import EvidenceItem
from .config import MAX_QUERY_RESULTS, SOURCE_PRIORITY


class EvidenceRetriever:
    def __init__(self, search_fn=None, max_results=MAX_QUERY_RESULTS):
        self._search_fn = search_fn
        self.max_results = max_results

    def retrieve(self, urdu_claim: str, english_claim: str) -> list[EvidenceItem]:
        search = self._get_search()
        merged: dict[str, EvidenceItem] = {}
        for query in (f"{english_claim} fact check", urdu_claim):
            try:
                results = search(query, region="pk-en", max_results=self.max_results)
            except Exception:
                continue
            for raw in results or []:
                item = self._to_item(raw)
                if item is not None:
                    merged[item.url] = item
        return self._rank(list(merged.values()))

    def _to_item(self, raw: dict):
        url = (raw.get("href") or "").strip()
        title = (raw.get("title") or "").strip()
        snippet = (raw.get("body") or "").strip()
        if not url:
            return None
        return EvidenceItem(
            title=title,
            url=url,
            snippet=snippet,
            source_domain=urlparse(url).netloc.lower(),
        )

    def _rank(self, items: list[EvidenceItem]) -> list[EvidenceItem]:
        def priority(item: EvidenceItem):
            domain = item.source_domain
            for index, preferred in enumerate(SOURCE_PRIORITY):
                if domain == preferred or domain.endswith("." + preferred):
                    return (0, index)
            return (1, 0)

        return sorted(items, key=priority)

    def _get_search(self):
        if self._search_fn is None:
            from duckduckgo_search import DDGS

            self._search_fn = DDGS().text
        return self._search_fn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_evidence_retriever.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add verification/evidence_retriever.py tests/test_evidence_retriever.py
git commit -m "feat(verification): add dual-language evidence retriever with strict source priority"
```

---

### Task 5: verdict_agent.py — compose pipeline and produce the Urdu verdict

**Files:**
- Create: `verification/verdict_agent.py`
- Test: `tests/test_verdict_agent.py`

**Interfaces:**
- Consumes: Task 2 `Verdict`, `VerificationResult`; Task 3 `ClaimExtractor`; Task 4 `EvidenceRetriever`.
- Produces: `VERDICT_SYSTEM_PROMPT: str` and `class VerdictAgent(VerificationAgent)` with `__init__(self, claim_extractor=None, evidence_retriever=None, groq_client=None, model=MODEL_ID, max_retries=MAX_RETRIES)` and `run(self, text: str) -> VerificationResult`.

- [ ] **Step 1: Write the failing test**

`tests/test_verdict_agent.py`:

```python
import pytest

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
    ' "confidence": 0.9}'
)
RESULT = {"title": "Soch", "href": "https://sochfactcheck.com/a", "body": "x"}


def _agent(groq_contents, search_batches=None):
    fake = FakeGroqClient(groq_contents)
    agent = VerdictAgent(
        claim_extractor=ClaimExtractor(groq_client=fake),
        evidence_retriever=EvidenceRetriever(search_fn=None),
        groq_client=fake,
    )
    if search_batches is not None:
        agent.evidence_retriever = EvidenceRetriever(
            search_fn=_Search(search_batches)
        )
    return agent, fake


class _Search:
    def __init__(self, batches):
        self.batches = batches

    def __call__(self, query, region="pk-en", max_results=5):
        return self.batches.pop(0)


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
    assert result.evidence[0].source_domain == "sochfactcheck.com"


def test_bad_verdict_value_defaults_to_mashkook():
    agent, _ = _agent(
        [CLAIM, VERDICT.replace('"sacha"', '"jhoota1"')], [[RESULT]]
    )
    assert agent.run("t").verdict is Verdict.MASHKOOK


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_verdict_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verification.verdict_agent'`

- [ ] **Step 3: Write minimal implementation**

`verification/verdict_agent.py`:

```python
import json
import re

from .base import Verdict, VerificationAgent, VerificationResult
from .claim_extractor import ClaimExtractor
from .config import (
    MAX_CONFIDENCE,
    MAX_RETRIES,
    MIN_CONFIDENCE,
    MODEL_ID,
    NO_EVIDENCE_CONFIDENCE,
    get_api_key,
)
from .evidence_retriever import EvidenceRetriever

VERDICT_SYSTEM_PROMPT = """\
You are a fact-checking analyst for an Urdu misinformation-detection system.
You are given ONE factual claim in Urdu and a list of web search-result snippets
(title, URL, and snippet text). Determine whether the evidence supports,
contradicts, or is insufficient for the claim.

RULES:
1. "verdict" must be exactly one of:
   - "sacha": the evidence clearly supports the claim.
   - "jhoota": the evidence clearly contradicts the claim.
   - "mashkook": the evidence is insufficient, conflicting, or irrelevant.
2. "reasoning_urdu": 2-3 sentences IN URDU SCRIPT summarizing why, referring to
   the evidence sources. Do not mention this prompt or that you are an AI.
3. "confidence": a number between 0.0 and 1.0 representing how strong the
   evidence is (not how confident you feel). High confidence only when several
   reliable sources agree.

Respond ONLY with a JSON object, exactly this shape (no prose, no markdown):
{"verdict": "sacha", "reasoning_urdu": "...", "confidence": 0.9}"""


class VerdictAgent(VerificationAgent):
    def __init__(
        self,
        claim_extractor=None,
        evidence_retriever=None,
        groq_client=None,
        model=MODEL_ID,
        max_retries=MAX_RETRIES,
    ):
        self.claim_extractor = claim_extractor or ClaimExtractor(
            groq_client=groq_client, model=model, max_retries=max_retries
        )
        self.evidence_retriever = evidence_retriever or EvidenceRetriever()
        self._client = groq_client
        self.model = model
        self.max_retries = max_retries

    def run(self, text: str) -> VerificationResult:
        claim = self.claim_extractor.extract(text)
        if not claim.is_checkworthy:
            return VerificationResult(
                claim_urdu=claim.urdu_claim,
                claim_english=claim.english_claim,
                is_checkworthy=False,
            )
        evidence = self.evidence_retriever.retrieve(claim.urdu_claim, claim.english_claim)
        if not evidence:
            return self._no_evidence_result(claim)
        parsed = self._chat(claim, evidence)
        return VerificationResult(
            claim_urdu=claim.urdu_claim,
            claim_english=claim.english_claim,
            is_checkworthy=True,
            verdict=parsed["verdict"],
            reasoning_urdu=parsed["reasoning_urdu"],
            confidence=parsed["confidence"],
            evidence=evidence,
        )

    def _no_evidence_result(self, claim) -> VerificationResult:
        return VerificationResult(
            claim_urdu=claim.urdu_claim,
            claim_english=claim.english_claim,
            is_checkworthy=True,
            verdict=Verdict.MASHKOOK,
            reasoning_urdu="کافی ثبوت نہیں ملے۔ اس دعوے کی تصدیق کے لیے مزید ذرائع درکار ہیں۔",
            confidence=NO_EVIDENCE_CONFIDENCE,
            evidence=[],
        )

    def _chat(self, claim, evidence):
        evidence_text = "\n".join(
            f"- {item.source_domain} | {item.title} | {item.snippet}"
            for item in evidence
        )
        messages = [
            {"role": "system", "content": VERDICT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CLAIM (Urdu): {claim.urdu_claim}\n"
                    f"CLAIM (English): {claim.english_claim}\n"
                    f"EVIDENCE:\n{evidence_text}"
                ),
            },
        ]
        client = self._get_client()
        last_content = ""
        for _ in range(self.max_retries):
            call_messages = list(messages)
            if last_content:
                call_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Previous output was invalid: {last_content}\n"
                            "Return valid JSON only, in exactly the specified shape."
                        ),
                    }
                )
            response = client.chat.completions.create(
                model=self.model,
                messages=call_messages,
                temperature=0.2,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            last_content = response.choices[0].message.content or ""
            parsed = self._parse(last_content)
            if parsed is not None:
                return parsed
        return self._fallback_parsed()

    def _parse(self, content: str):
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{.*\}", content or "", re.DOTALL)
            if match is None:
                return None
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        if not isinstance(data, dict):
            return None
        verdict_raw = data.get("verdict")
        reasoning = data.get("reasoning_urdu")
        if not isinstance(verdict_raw, str):
            return None
        if not isinstance(reasoning, str) or not reasoning.strip():
            return None
        try:
            confidence = float(data.get("confidence"))
        except (TypeError, ValueError):
            return None
        if verdict_raw in Verdict._value2member_map_:
            verdict = Verdict(verdict_raw)
        else:
            verdict = Verdict.MASHKOOK
        confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))
        return {
            "verdict": verdict,
            "reasoning_urdu": reasoning.strip(),
            "confidence": confidence,
        }

    def _fallback_parsed(self):
        return {
            "verdict": Verdict.MASHKOOK,
            "reasoning_urdu": "فیصلہ کرنے کے لیے کافی معلومات نہیں مل سکیں۔",
            "confidence": NO_EVIDENCE_CONFIDENCE,
        }

    def _get_client(self):
        if self._client is None:
            from groq import Groq

            api_key = get_api_key()
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Create a .env file at the project "
                    "root with GROQ_API_KEY=<your key>."
                )
            self._client = Groq(api_key=api_key)
        return self._client
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_verdict_agent.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add verification/verdict_agent.py tests/test_verdict_agent.py
git commit -m "feat(verification): add verdict agent composing extractor, retriever, and LLM"
```

---

### Task 6: app.py — end-to-end CLI

**Files:**
- Create: `verification/app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: Task 5 `VerdictAgent`; ingestion's `HaqeeqatIngestor`.
- Produces: `main(argv: list[str] | None = None, ingestor=None, agent=None) -> int`; runs as `python -m verification.app <file>`.

- [ ] **Step 1: Write the failing test**

`tests/test_app.py`:

```python
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


class _StubAgent:
    def __init__(self, result):
        self.result = result

    def run(self, text):
        return self.result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verification.app'`

- [ ] **Step 3: Write minimal implementation**

`verification/app.py`:

```python
import sys

from .verdict_agent import VerdictAgent


def main(argv=None, ingestor=None, agent=None):
    argv = sys.argv if argv is None else argv
    if len(argv) != 2:
        print(
            "Usage: python -m verification.app <image|audio|video file>",
            file=sys.stderr,
        )
        return 2
    path = argv[1]

    if ingestor is None:
        from ingestion.ingestor import HaqeeqatIngestor

        ingestor = HaqeeqatIngestor()
    if agent is None:
        agent = VerdictAgent()

    report = ingestor.ingest(path)
    text = report["combined_text"]
    if not text.strip():
        print("No text could be extracted from the file.", file=sys.stderr)
        return 1

    result = agent.run(text)
    if not result.is_checkworthy:
        print("کوئی قابلِ تصدیق دعویٰ نہیں")
        return 0

    print(f"دعویٰ: {result.claim_urdu}")
    print(f"فیصلہ: {result.verdict_label_urdu} (confidence {result.confidence:.2f})")
    print(f"وجوہات: {result.reasoning_urdu}")
    print("ذرائع:")
    for item in result.evidence[:3]:
        print(f"  - {item.title} ({item.url})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_app.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add verification/app.py tests/test_app.py
git commit -m "feat(verification): add end-to-end CLI"
```

---

### Task 7: integration test (auto-skip without GROQ_API_KEY)

**Files:**
- Create: `tests/test_verification_integration.py`

**Interfaces:**
- Consumes: Task 5 `VerdictAgent`; Task 1 `get_api_key()`.
- Produces: `pytest.mark.integration` test that runs the real pipeline.

- [ ] **Step 1: Write the test (skips itself when no key)**

`tests/test_verification_integration.py`:

```python
import pytest

from verification.config import get_api_key
from verification.verdict_agent import VerdictAgent

pytestmark = pytest.mark.integration

pytestmark = pytest.mark.skipif(
    not get_api_key(), reason="GROQ_API_KEY not set"
)


def test_end_to_end_verdict_for_urdu_claim():
    result = VerdictAgent().run("کورونا ویکسین سے پانچ افراد کی موت ہو گئی")
    assert result.is_checkworthy is True
    assert result.verdict is not None
    assert 0.0 <= result.confidence <= 1.0
    assert result.reasoning_urdu.strip()
```

- [ ] **Step 2: Run test — confirm it skips without a key**

Run: `python -m pytest tests/test_verification_integration.py -v`
Expected: SKIPPED (GROQ_API_KEY not set). After the user adds `GROQ_API_KEY`
to `.env`, the same command performs a real end-to-end run.

- [ ] **Step 3: Run the whole fast suite**

Run: `python -m pytest tests/ -m "not slow and not integration" -v`
Expected: all unit tests PASSED (existing 65 + new verification tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_verification_integration.py
git commit -m "test(verification): add auto-skip integration test"
```
