# Haqeeqat Check — Verification & Reasoning Engine Design

Date: 2026-08-12

## Purpose

Second module of **Haqeeqat Check**, an Urdu/English misinformation detector.
Consumes the `combined_text` produced by the ingestion module and returns a
verified Urdu verdict: does the text contain a check-worthy factual claim, and
is that claim **سچا (true)**, **جھوٹا (false)**, or **مشکوک (unverifiable)**?

## Non-Goals

- No image/audio/video re-analysis; this module works on text only.
- No page-level web scraping; evidence is search-result snippets only.
- No persistent storage of evidence or verdicts (stateless pipeline).
- No sentiment / opinion / disinformation-style analysis of non-factual text.

## Output Contract

`VerificationAgent.run(combined_text: str) -> VerificationResult`:

```python
@dataclass
class EvidenceItem:
    title: str
    url: str
    snippet: str
    source_domain: str      # e.g. "sochfactcheck.com"

@dataclass
class SearchableClaim:
    is_checkworthy: bool
    urdu_claim: str         # ONE Urdu-script sentence (Roman Urdu converted to Urdu script)
    english_claim: str      # faithful English translation for web search

@dataclass
class VerificationResult:
    claim_urdu: str
    claim_english: str
    is_checkworthy: bool
    verdict: Verdict | None    # None when is_checkworthy is False
    verdict_label_urdu: str    # "سچا" | "جھوٹا" | "مشکوک"
    reasoning_urdu: str
    confidence: float          # 0.0-1.0, evidence strength, not model certainty
    evidence: list[EvidenceItem]
```

`Verdict` is a `str`-enum:

```python
class Verdict(str, enum.Enum):
    SACHA     = "sacha"     # supported by evidence
    JHOOOTA   = "jhoota"    # contradicted by evidence
    MASHKOOK  = "mashkook"  # insufficient or conflicting evidence
```

## Architecture

Pipeline of three agents, each an isolated callable, wired together by a CLI:

```
combined_text
   │
   ▼
ClaimExtractorAgent ──(Groq Llama-3.3-70B)──► SearchableClaim
   │                                              │
   ▼                                              │ urdu_claim + english_claim
EvidenceRetrieverAgent ──(duckduckgo-search)──►   │
   │                                              ▼
   │ evidence: list[EvidenceItem]            VerdictAgent ──(Groq)──► VerificationResult
   └──────────────────────────────────────────────►
```

```
verification/
├── __init__.py              # exports Verdict, VerificationResult, agents
├── base.py                  # Verdict enum, URDU_LABELS, dataclasses, VerificationAgent ABC
├── config.py                # env key, model id, source priorities, prompt constants
├── claim_extractor.py       # ClaimExtractorAgent
├── evidence_retriever.py    # EvidenceRetrieverAgent
├── verdict_agent.py         # VerdictAgent
└── app.py                   # end-to-end CLI (Phase 4)
```

### base.py
- `class Verdict(str, enum.Enum)` — canonical verdict values.
- `URDU_LABELS = {"sacha": "سچا", "jhoota": "جھوٹا", "mashkook": "مشکوک"}`.
- Dataclasses `EvidenceItem`, `SearchableClaim`, `VerificationResult`.
- `class VerificationAgent(ABC)` with `run(self, text: str) -> VerificationResult`.
  All agents accept optional injected dependencies (Groq client, DDG function)
  so tests never touch the network.

### config.py
- Loads `.env` via `python-dotenv` (from CWD and repo root); reads
  `GROQ_API_KEY`.
- `MODEL_ID = "llama-3.3-70b-versatile"`.
- `SOURCE_PRIORITY` list — strict, first match wins when ranking:
  1. `sochfactcheck.com`
  2. `afp.com`
  3. `dawn.com`
  (any other domain is lower priority than these three)
- Prompt template + parsing constants shared by extractor/verdict agents.

### claim_extractor.py
- `ClaimExtractorAgent(verification_agent)` — extracts the single most
  check-worthy claim from noisy text (OCR + transcript).
- Calls Groq `chat.completions.create` with `model=MODEL_ID`,
  `temperature=0.2`, `max_tokens=300`, `response_format={"type":
  "json_object"}`.
- **Roman Urdu → Urdu script:** the prompt explicitly instructs the model that
  the input may be written in Roman Urdu (Urdu in English/Latin script) and
  that `urdu_claim` must ALWAYS be emitted in proper Urdu script
  (اردو رسم الخط), transliterating from Roman Urdu where needed.
- **Retry/validation loop (max 3):** parse JSON → validate keys/types →
  on failure resend with the previous bad output as feedback
  ("Previous output was invalid: <reason>. Return valid JSON only.").
- Non-checkworthy text or exhausted retries → `SearchableClaim(
  is_checkworthy=False)` with empty claims.

System prompt (English for best JSON compliance; Urdu lives in the data):

```
You are the claim-extraction step of an Urdu fact-checking pipeline.
You receive raw, noisy text (OCR from images, or a speech-to-text transcript).
The text may be written in Urdu script or in Roman Urdu (Urdu written in
English/Latin letters). Decide whether it contains ONE verifiable factual
claim, and if so extract it.

RULES:
1. A claim is check-worthy ONLY if it asserts a verifiable fact about the real
   world: politics, economy, health, religion, sports, crime, viral rumors, etc.
2. IGNORE as not check-worthy: greetings, pleasantries, personal opinions,
   poetry, personal stories/experiences, pure questions, jokes, and content
   with no factual assertion.
3. If several claims exist, extract only the SINGLE most viral / important one.
4. "urdu_claim": the claim as ONE concise sentence, ALWAYS written in proper
   Urdu script (اردو رسم الخط). If the original is in Roman Urdu, transliterate
   it into Urdu script. Never add your own interpretation.
5. "english_claim": a faithful English translation of urdu_claim, written for
   international web search (e.g. mention "Pakistan" if the claim concerns it).
6. "is_checkworthy": true ONLY if a verifiable claim passes rules 1-5.

Respond ONLY with a JSON object, exactly this shape (no prose, no markdown):
{"is_checkworthy": true, "urdu_claim": "...", "english_claim": "..."}
```

User message wraps the text in `<text>...</text>` delimiters so the model
cannot confuse instructions with content.

### evidence_retriever.py
- `EvidenceRetrieverAgent(search_fn=None)` — wraps `duckduckgo_search.DDGS`.
- **Dual-language search** (approved enhancement):
  1. `DDGS().text(f"{english_claim} fact check", region="pk-en", max_results=5)`
  2. `DDGS().text(f"{urdu_claim}", region="pk-en", max_results=5)`
- **Merge + dedupe by URL** — later duplicates replace earlier ones, keyed on
  normalized `url`.
- **Strict source prioritization** — sort so that any result from
  `sochfactcheck.com` / `afp.com` / `dawn.com` ranks above results from other
  domains (stable within the same priority tier, source_order preserved).
- `RateLimitException` (from `duckduckgo_search`) → degrade to empty evidence,
  never crash. Malformed results (missing url/snippet) are dropped.
- Returns `list[EvidenceItem]` sorted by priority.

### verdict_agent.py
- `VerdictAgent(claim_extractor?, evidence_retriever?, groq_client=None)`.
  The composing agent: runs extractor → retriever → LLM verdict.
- LLM call returns JSON `{"verdict": "sacha|jhoota|mashkook",
  "reasoning_urdu": "...", "confidence": 0.85}`.
- System prompt: fact-checker role; verdict meanings (سچا = supported,
  جھوٹا = contradicted, مشکوک = insufficient/conflicting); reasoning must be
  2–3 Urdu sentences citing the evidence sources; confidence reflects
  evidence strength (0.0–1.0), NOT model self-certainty.
- Validation: verdict must be a valid enum value (else default `mashkook`),
  confidence clamped to `[0.0, 1.0]`; same retry loop as extractor.
- **Fallback:** empty evidence → verdict `mashkook`, confidence ≈ 0.3,
  reasoning = "کافی ثبوت نہیں ملے" — never fabricate a verdict.

### app.py (Phase 4 CLI)
`python -m verification.app <file>`:
1. `HaqeeqatIngestor.ingest(file)` → abort with message if `combined_text` empty.
2. Extract claim → if `is_checkworthy=False`: print "کوئی قابلِ تصدیق دعویٰ نہیں"
   and exit 0.
3. Retrieve evidence → pass to verdict agent.
4. Print: Urdu claim, verdict label (سچا/جھوٹا/مشکوک), reasoning, confidence,
   top 3 sources with URLs.

## Error Handling

- `GROQ_API_KEY` missing → clear actionable error: "Set GROQ_API_KEY in .env".
- Groq API errors (auth, rate limit, network) → raise with a friendly wrapper
  message; never silently produce a verdict.
- DDG rate limit → empty evidence, still reach a `mashkook` fallback.
- Bad LLM JSON after 3 retries → `SearchableClaim(is_checkworthy=False)` for
  extraction; `mashkook` fallback for verdict.

## Dependencies

`requirements.txt` additions: `groq`, `duckduckgo-search`, `python-dotenv`.
Python 3.12.

## Testing

- **Unit (pytest, offline, fast):** all LLM/DDG calls stubbed via injected
  fakes — no network, no real key.
  - claim_extractor: valid/invalid/partial JSON, retry-on-invalid, non-
    checkworthy text, Roman-Urdu prompt contains transliteration instruction.
  - evidence_retriever: dual query construction, URL dedupe, strict domain
    priority (sochfactcheck > afp > dawn > other), rate-limit → empty.
  - verdict_agent: enum parsing, confidence clamp, no-evidence fallback,
    `is_checkworthy=False` short-circuit.
- **Integration (slow, auto-skip unless `GROQ_API_KEY` set):** real DDG +
  real Groq end-to-end on a sample claim.
- Existing 65-test fast suite must keep passing.

## Sequence of Work

1. Scaffold package, `requirements.txt`, `config.py`
2. `base.py` (enum, dataclasses, ABC)
3. `claim_extractor.py` + tests
4. `evidence_retriever.py` + tests
5. `verdict_agent.py` + tests
6. `app.py` CLI + end-to-end test
7. Integration test (auto-skip)
8. Manual smoke test with `.env` key
