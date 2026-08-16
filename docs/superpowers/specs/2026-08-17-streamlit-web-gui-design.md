# Streamlit Web GUI for Haqeeqat Check — Design

Date: 2026-08-17
Status: Approved
Scope: A public Hugging Face Spaces web interface for Haqeeqat Check with a
bilingual (Urdu + English) verdict presentation.

## Goal

Let anyone on the web run a fact-check by pasting Urdu text or uploading
media (jpg/png/mp3/mp4/wav), and see the verdict, reasoning, and sources in
both Urdu and English. The Space is a public demo; the free-tier CPU must not
be forced to load OCR/Whisper models unless media is actually uploaded.

## Architecture

- `app_gui.py` at the repo root, run via `streamlit run app_gui.py`.
- Dependency-injected `main(st=None, ingestor=None, agent=None)` mirroring
  `verification/app.py`. Lazy imports of streamlit and heavy modules inside
  `main()` so pytest collection stays fast and importing the module never
  loads torch/whisper.
- Radio selector:
  - "Paste text / متن چیک کریں": `st.text_area` + check button → `agent.run(text)`.
    Instant; no models.
  - "Upload media / میڈیا اپ لوڈ کریں": `st.file_uploader` (jpg, png, mp3,
    mp4, wav) → write to a temp file → `_ensure_models()` (downloads the
    UTRNet/YOLO weights via `tools/download_urdu_models.py` on first media
    check, models stay gitignored) → `HaqeeqatIngestor().ingest(path)` →
    `agent.run(combined_text)`.
- Agent is `VerdictAgent` (the concrete composing agent in this repo).
- Groq key resolution: `os.environ["GROQ_API_KEY"]` first (HF Spaces secrets
  are env vars); fall back to `st.secrets.get("GROQ_API_KEY")`, setting the
  env var so `config.get_api_key()` works unchanged.

## Bilingual verdict (core agent extension)

- `verification/base.py`: add `reasoning_english: str = ""` to
  `VerificationResult`.
- `verification/verdict_agent.py`: the system prompt asks for both
  `reasoning_urdu` and `reasoning_english` (same content, Urdu script and
  English); JSON shape gains `reasoning_english`. `_parse` fills it,
  falling back to the Urdu text if the model omits it. `_fallback_parsed`
  and `_no_evidence_result` gain English strings.
- Verdict labels map `sacha/jhoota/mashkook` → `سچا/جھوٹا/مشکوک` and
  `True/False/Unverified`.

## UI rendering

- Title: "Haqeeqat Check: Urdu Misinformation Detector"; subtitle
  "حقیقت چیک: اردو غلط معلومات کی جانچ پڑتال".
- Spinner "Processing Media..." around ingest/run for media; the extracted
  text appears in an expander (`استخراج شدہ متن / Extracted text`).
- Empty extracted text → `st.warning` + stop. Not checkworthy → `st.info`.
- Verdict color boxes (streamlit alert components): `سچا/True` →
  `st.success` (green), `جھوٹا/False` → `st.error` (red),
  `مشکوک/Unverified` → `st.warning` (yellow), each showing confidence.
- Claim and reasoning render bilingually via a small injected `<style>`
  block: Urdu in RTL, Nastaliq font stack, ~22px; English below it.
- Evidence in the sidebar (`شواہد / Sources`): title as a link, snippet.

## Deployment

- `requirements.txt`: add `streamlit>=1.33.0`.
- New `README.md` (repo root): Hugging Face Spaces frontmatter
  (`sdk: streamlit`, `app_file: app_gui.py`, title, emoji, colorFrom/To)
  plus a short English + Urdu description, local run instructions, and the
  GROQ_API_KEY secrets note.
- Models are NOT committed: `_ensure_models()` streams the two weight files
  (~95 MB) from the existing HF URL on first media check.

## Testing

- `tests/test_app_gui.py`: a recording fake `st` (radio/text_area/button/
  file_uploader/spinner/expander/alert boxes/sidebar/secrets/stop), a fake
  ingestor and fake agent. Cases:
  - sacha→success, jhoota→error, mashkook→warning, not-checkworthy→info;
  - text mode passes the text to `agent.run`;
  - media mode writes a temp file with the right suffix and calls
    `ingestor.ingest`, and triggers `_ensure_models`;
  - evidence items render in the sidebar with URLs;
  - bilingual claim + reasoning render in one markdown block;
  - `st.secrets` fallback populates `os.environ["GROQ_API_KEY"]`.
- `tests/test_verdict_agent.py`: fixture `VERDICT` gains `reasoning_english`;
  assert `result.reasoning_english` is populated.

## Out of scope

- Translating the raw extracted OCR text to English (extractor already
  produces `english_claim`; the verdict/reasoning are bilingual).
- Whisper/OCR model hosting (downloaded at runtime, not committed).
