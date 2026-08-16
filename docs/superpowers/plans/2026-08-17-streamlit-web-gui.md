# Streamlit Web GUI — Implementation Plan

Base: design doc `docs/superpowers/specs/2026-08-17-streamlit-web-gui-design.md`

## Steps

1. **Core agent bilingual** (`verification/base.py`, `verification/verdict_agent.py`)
   - Add `reasoning_english` to `VerificationResult`.
   - Prompt rule 2b + JSON shape; `_parse`/`_fallback_parsed`/`_no_evidence_result` English.
2. **`app_gui.py`** — DI `main(st, ingestor, agent)`, text + media modes,
   `_ensure_api_key`, `_ensure_models`, `_render_verdict`, `_render_evidence`,
   `_inject_css`, `_bilingual_block`.
3. **`requirements.txt`** — add `streamlit>=1.33.0`.
4. **`README.md`** — HF Spaces frontmatter + blurb.
5. **Tests** — update `tests/test_verdict_agent.py` fixtures; new
   `tests/test_app_gui.py` with recording fake `st`.
6. **Verify** — `pytest` fast suite green; manual `streamlit run app_gui.py`
   smoke (text path + one media check).

## Verification commands

- `.\\.venv\\Scripts\\python.exe -m pytest -m "not slow and not integration" -q`
- `streamlit run app_gui.py` (manual smoke; requires GROQ_API_KEY)
