---
title: Haqeeqat Check
emoji: ✔️
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.33.0
app_file: app_gui.py
pinned: false
---

# Haqeeqat Check — Urdu Misinformation Detector

**حقیقت چیک — اردو غلط معلومات کی جانچ پڑتال**

Upload a screenshot, audio clip, or video in Urdu, or paste Urdu text directly,
and Haqeeqat Check will extract the claim, search for evidence, and return a
verdict (سچا / True, جھوٹا / False, مشکوک / Unverified) with sources — in both
Urdu and English.

- **Text check** (instant, no model download): paste an Urdu sentence and press
  *Check*.
- **Media check**: upload a `jpg`, `png`, `mp3`, `mp4`, or `wav` file. On the
  first media check the OCR/transcription models (~95 MB) are downloaded
  automatically; afterwards they are cached for the session.

## Run locally

```bash
pip install -r requirements.txt
python tools/download_urdu_models.py   # only needed for media uploads
streamlit run app_gui.py
```

## API key

Add `GROQ_API_KEY` in **Settings → Secrets** on the Space (or in a local
`.env` file). The app reads it from the environment and falls back to
`st.secrets`.

## How it works

1. **Ingest** (`HaqeeqatIngestor`) — UTRNet/YOLO Urdu OCR with a PaddleOCR
   fallback for on-screen text, Whisper for speech.
2. **Extract** (`ClaimExtractor`) — pulls the checkworthy claim and translates
   it to English (attributed quotes by public figures are treated as claims).
3. **Retrieve** (`EvidenceRetriever`) — searches the web for fact checks.
4. **Verdict** (`VerdictAgent`) — a Groq LLM (llama-3.3-70b) weighs the
   evidence and returns the verdict, confidence, and bilingual reasoning.
