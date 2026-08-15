# Attributed-Quote Checkworthiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the claim extractor treat statements attributed to a named public figure as check-worthy attribution claims, so viral signed quotes like "…! — مریم نواز" reach retrieval and verdict instead of being skipped as poetry.

**Architecture:** Add a pure `detect_attribution(text) -> str | None` function to `verification/claim_extractor.py` that recognises explicit, signature, and colon-quote attribution structures. Wire it into `ClaimExtractor.extract()` so a detected attribution appends a hint to the model instructions and forces `is_checkworthy=True`. Add an exception rule to `CLAIM_EXTRACTION_SYSTEM_PROMPT`. Downstream retrieval/verdict are unchanged.

**Tech Stack:** Python 3.12, stdlib `re`, Groq (llama-3.3-70b-versatile) via the existing `ClaimExtractor`, pytest with `tests/fakes.py::FakeGroqClient`.

## Global Constraints

- No NER dependency; the detector recognises attribution *structure* only; the model decides the actual name.
- No curated public-figure list.
- Do NOT modify `evidence_retriever.py`, `verdict_agent.py`, `app.py`, or `base.py`.
- The attribution candidate is a hint only, never trusted as truth.
- False positives are acceptable — over-checking beats missing a viral quote.
- Unit tests are offline via `FakeGroqClient`; the integration test auto-skips when `GROQ_API_KEY` is unset.
- Run tests with the venv interpreter: `.\.venv\Scripts\python.exe -m pytest <path> -q`
- Commit only the files listed in each task (Module 1 scratchpad stays uncommitted).

---

### Task 1: `detect_attribution` — attribution structure detector

**Files:**
- Modify: `verification/claim_extractor.py` (add `import re` at top, add `detect_attribution` function + module-level regexes before `class ClaimExtractor`)
- Test: `tests/test_claim_extractor.py` (add import + tests)

**Interfaces:**
- Produces: `detect_attribution(text: str) -> str | None` — returns a candidate name string when the text exhibits attribution structure, else `None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_claim_extractor.py`:

```python
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
    text = "میرا بس چلے\nمیرا بس چلے تو میں پیتہ نہیں\nآپ کو کیا کیا دے دوں! مریم نوز"
    assert detect_attribution(text) == "مریم نوز"


def test_detects_colon_quote_prefix():
    assert detect_attribution('مریم نواز: "یہ سب جھوٹ ہے"') == "مریم نواز"


def test_returns_none_for_plain_news_text():
    assert detect_attribution("سندھ میں بارش سے تین افراد ہلاک ہو گئے") is None


def test_returns_none_for_greeting():
    assert detect_attribution("السلام علیکم") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_claim_extractor.py -q`
Expected: FAIL — `ImportError: cannot import name 'detect_attribution'`.

- [ ] **Step 3: Implement `detect_attribution`**

In `verification/claim_extractor.py`, change the import block:

```python
import json
import re
```

Add these module-level regexes and the function just before `class ClaimExtractor:`:

```python
_URDU_EXPLICIT_RE = re.compile(
    r"([\u0600-\u06FF][\u0600-\u06FF\s]{2,40}?)\s+"
    r"(?:نے\s+کہا|کہتی\s+ہیں|کہتے\s+ہیں|کہا\s+کہ)"
)
_ENGLISH_EXPLICIT_RE = re.compile(
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\s+"
    r"(?:said|says|stated|told|claimed)\b"
)
_COLON_QUOTE_RE = re.compile(
    r"^([A-Za-z\u0600-\u06FF][\w\u0600-\u06FF\s]{1,40}):\s*[\"\u201c«]",
    re.MULTILINE,
)
_SIGNATURE_RE = re.compile(
    r"(?:—|–|-|!|۔)\s*([\w\u0600-\u06FF][\w\u0600-\u06FF\s]{1,40})\s*$"
)


def detect_attribution(text: str) -> str | None:
    """Return a candidate name if ``text`` exhibits attribution structure.

    Recognises explicit verbal attribution ("X said", "X نے کہا"), a signed
    quote ("— X", "…! X") on the last non-empty line, and a colon-quote
    prefix ("X: "). The candidate is a hint only, never trusted as truth.
    """
    explicit = _URDU_EXPLICIT_RE.search(text or "") or _ENGLISH_EXPLICIT_RE.search(
        text or ""
    )
    if explicit:
        return explicit.group(1).strip()
    colon = _COLON_QUOTE_RE.search(text or "")
    if colon:
        return colon.group(1).strip()
    for line in reversed((text or "").splitlines()):
        if line.strip():
            signature = _SIGNATURE_RE.search(line.strip())
            if signature:
                return signature.group(1).strip()
            return None
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_claim_extractor.py -q`
Expected: PASS (all existing + 7 new tests).

- [ ] **Step 5: Commit**

```bash
git add verification/claim_extractor.py tests/test_claim_extractor.py
git commit -m "feat: detect attributed-quote structure in claim text"
```

---

### Task 2: Wire the detector into `ClaimExtractor.extract()` + prompt rule

**Files:**
- Modify: `verification/claim_extractor.py` (`CLAIM_EXTRACTION_SYSTEM_PROMPT`, `ClaimExtractor.extract`, add `_user_content`)
- Test: `tests/test_claim_extractor.py`

**Interfaces:**
- Consumes: `detect_attribution(text: str) -> str | None` (Task 1).
- Produces: `ClaimExtractor.extract()` now forces `is_checkworthy=True` and appends a hint to the user message when attribution is detected. `SearchableClaim` shape unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_claim_extractor.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_claim_extractor.py -q`
Expected: FAIL — all 3 new tests (prompt lacks the rule, no override, no hint).

- [ ] **Step 3: Update the system prompt**

In `verification/claim_extractor.py`, replace rule 2 in `CLAIM_EXTRACTION_SYSTEM_PROMPT` with rule 2 + new rule 2b:

```
2. IGNORE as not check-worthy: greetings, pleasantries, personal opinions,
   poetry, personal stories or experiences, pure questions, jokes, and any
   content with no factual assertion.
2b. EXCEPTION (attributed quotes): a statement presented as the words of a
   named public figure — explicit attribution ("X said", "X نے کہا"), a signed
   quote ("— X", "…! X"), or "X: ..." — IS check-worthy even if poetic,
   lyrical, or humorous, because "did X actually say this?" is a verifiable
   fact. Extract such claims as attribution questions: urdu_claim "کیا <X> نے
   یہ کہا: ...؟" and english_claim "Did <X> say: '...'?" preserving the exact
   wording of the statement.
```

- [ ] **Step 4: Update `extract()` and add `_user_content`**

In `verification/claim_extractor.py`, replace the `extract` method and add `_user_content`:

```python
    def extract(self, text: str) -> SearchableClaim:
        name = detect_attribution(text)
        messages = [
            {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": self._user_content(text, name)},
        ]
        parsed = self._chat(messages)
        if parsed is None:
            return SearchableClaim(is_checkworthy=False)
        return SearchableClaim(
            is_checkworthy=parsed["is_checkworthy"] or name is not None,
            urdu_claim=parsed["urdu_claim"],
            english_claim=parsed["english_claim"],
        )

    def _user_content(self, text: str, name: str | None) -> str:
        content = f"<text>\n{text}\n</text>"
        if name:
            content += (
                '\n\nHINT: the text appears to present a statement as the words '
                'of a person (possible name: "%s"). Extract the claim as an '
                'attribution question: "کیا <person> نے یہ کہا؟" / '
                '"Did <person> say this?", preserving the exact wording of the '
                "statement." % name
            )
        return content
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_claim_extractor.py -q`
Expected: PASS (existing tests including `test_input_is_wrapped_in_text_delimiters` still pass — plain text gets no hint).

- [ ] **Step 6: Commit**

```bash
git add verification/claim_extractor.py tests/test_claim_extractor.py
git commit -m "feat: treat attributed quotes as checkworthy in claim extractor"
```

---

### Task 3: Slow integration test on the real lyric

**Files:**
- Modify: `tests/test_verification_integration.py`

**Interfaces:**
- Consumes: `VerdictAgent().run(text)` (unchanged), real Groq + DDG.
- Produces: verification that the full pipeline now returns a check-worthy, verdict-carrying result for the actual `UrduTest1.jpg` lyric text.

- [ ] **Step 1: Add the integration test**

Append to `tests/test_verification_integration.py`:

```python
def test_attributed_lyric_is_checkworthy():
    lyric = "میرا بس چلے\nمیرا بس چلے تو میں پیتہ نہیں\nآپ کو کیا کیا دے دوں! مریم نوز"
    result = VerdictAgent().run(lyric)
    assert result.is_checkworthy is True
    assert result.verdict is not None
```

The module already carries `pytestmark` (`integration` marker + `skipif` without `GROQ_API_KEY`), so no extra setup is needed.

- [ ] **Step 2: Run the integration test**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_verification_integration.py -m integration -v`
Expected: PASS — `is_checkworthy` is `True`, verdict present (likely مشکوک).

- [ ] **Step 3: Run the full fast suite**

Run: `.\.venv\Scripts\python.exe -m pytest -q -m "not integration and not slow"`
Expected: PASS — all fast tests, 4 deselected.

- [ ] **Step 4: Commit**

```bash
git add tests/test_verification_integration.py
git commit -m "test: attributed lyric reaches verdict end-to-end"
```

---

## Optional Manual Smoke Test

Run the CLI on the real image to confirm the fix end-to-end:

```powershell
cd C:\Users\hp\Documents\HaqeeqatCheck
.\.venv\Scripts\python.exe -X utf8 -m verification.app UrduTest1.jpg
```

Expected: no longer prints "کوئی قابلِ تصدیق دعویٰ نہیں" — it prints a دعویٰ framed as an attribution question, a فیصلہ, وجوہات, and ذرائع.
