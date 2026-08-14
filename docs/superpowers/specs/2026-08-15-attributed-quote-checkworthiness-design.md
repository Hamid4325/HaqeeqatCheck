# Haqeeqat Check — Attributed-Quote Checkworthiness Design

Date: 2026-08-15

## Purpose

Fix a miss in the verification engine's claim-extraction step. When a piece of
text is presented as the words of a **named public figure** — an explicit
attribution ("X said…"), a signed poster ("…! — Maryam Nawaz"), or a colon
quote ("X: …") — it is a verifiable attribution claim even if the wording is
poetic/lyrical. Fact-checkers routinely verify such quotes ("did X actually say
this?"). Today the extractor skips them as "poetry", so a viral attributed
quote never reaches retrieval or the verdict stage.

Example failure: OCR of `UrduTest1.jpg` yields
`میرا بس چلے / میرا بس چلے تو میں پیتہ نہیں / آپ کو کیا کیا دے دوں! مریم نوز`
(a satirical lyric signed "— Maryam Nawaz"). The extractor returns
`is_checkworthy=False` and the app prints "کوئی قابلِ تصدیق دعویٰ نہیں".

## Non-Goals

- No new retrieval/verdict logic: `EvidenceRetriever` already searches
  `f"{english_claim} fact check"` plus the Urdu claim, which suits
  attribution questions.
- No curated figure list; any named public figure counts.
- No NER dependency; the detector recognises *attribution structure* only, and
  the model decides the actual name.

## Design

Two coordinated changes in `verification/claim_extractor.py`, plus a prompt
rule.

### 1. Attribution detector — pure function

```python
def detect_attribution(text: str) -> str | None
```

Returns a candidate name string when the text exhibits attribution structure,
else `None`. Patterns (applied to the text as a whole):

- **Explicit verbal attribution** — regex on:
  - Urdu: `نے کہا`, `کہتی ہیں`, `کہتے ہیں`, `کہا کہ`, `کے مطابق`, `کا قول`
  - English: `said`, `says`, `stated`, `told`, `claimed`
- **Signature-style attribution** — the last non-empty line ends with
  `— Name`, `- Name`, or `…! Name`. The captured trailing tokens are the
  candidate name (OCR noise tolerated, e.g. `مریم نوز`).
- **Colon-quote prefix** — a line starting `Name: "…"` or `Name: «…»`.

The candidate is a *hint only*, never trusted as truth.

### 2. `ClaimExtractor.extract()` wiring

- Run `detect_attribution(text)` first.
- If it fires: append an attribution hint to the extraction instructions —
  "the text appears to present a statement as the words of a person (possible
  name: `<candidate>`); extract the claim as an attribution question:
  `کیا <person> نے یہ کہا؟` / `Did <person> say this?`, preserving the exact
  wording of the statement."
- **Override `is_checkworthy=True`** after a successful parse so the claim
  cannot be skipped as poetry.
- If the model parse fails → existing fallback `SearchableClaim(
  is_checkworthy=False)` (unchanged).

### 3. System prompt rule

Add to `CLAIM_EXTRACTION_SYSTEM_PROMPT`: a statement presented as a named
public figure's words (signed or attributed quote) is check-worthy even if it
is poetic/lyrical; the extracted claim becomes an attribution question.

## Behavior — false positives

A non-quote line ending like `…! شکر ہے` could trip the `! Name` pattern and
force a check. Trade-off accepted: better to over-check (low-confidence
مشکوک verdict) than miss a viral attributed quote. The model still writes the
actual claim text, bounding the damage.

## Testing

- **Unit (pytest, offline, fast):**
  - `detect_attribution`: Urdu explicit (`مریم نواز نے کہا…`), English
    explicit (`Maryam Nawaz said…`), dash signature (`— Maryam Nawaz`),
    `! Name` OCR-noisy (`…! مریم نوز`), colon prefix (`مریم نواز: "…"`), and
    non-matches (plain news text, greeting).
  - `extract()` with a fake Groq client: attribution present but model returns
    `is_checkworthy=False` → result is checkworthy and the hint was included
    in the messages.
  - Existing claim-extractor tests keep passing.
- **Integration (slow, auto-skip unless `GROQ_API_KEY` set):** real run on the
  actual lyric text → `is_checkworthy=True`.

## Files Changed

- `verification/claim_extractor.py` — detector + wiring + prompt rule.
- `tests/test_claim_extractor.py` — new detector + extract-override tests.
- `tests/test_verification_integration.py` — optional slow attribution test.
