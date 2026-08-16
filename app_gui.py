"""Streamlit web interface for Haqeeqat Check (hosted on Hugging Face Spaces).

Run locally with:  streamlit run app_gui.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

_URDU_LABELS = {
    "sacha": "سچا",
    "jhoota": "جھوٹا",
    "mashkook": "مشکوک",
}
_ENGLISH_LABELS = {
    "sacha": "True",
    "jhoota": "False",
    "mashkook": "Unverified",
}
_BOXES = {"sacha": "success", "jhoota": "error", "mashkook": "warning"}
_ICONS = {"sacha": "✔️", "jhoota": "❌", "mashkook": "⚠️"}
_MODEL_FILES = ["best_norm_ED.pth", "yolov8m_UrduDoc.pt"]


def main(st=None, ingestor=None, agent=None):
    if st is None:
        import streamlit as st

    st.set_page_config(
        page_title="Haqeeqat Check: Urdu Misinformation Detector",
        page_icon="✔️",
        layout="wide",
    )
    st.title("Haqeeqat Check: Urdu Misinformation Detector")
    st.markdown("##### حقیقت چیک: اردو غلط معلومات کی جانچ پڑتال")
    _inject_css(st)
    _ensure_api_key(st)

    mode = st.radio(
        "Input method / ان پٹ کا طریقہ",
        ["Paste text / متن چیک کریں", "Upload media / میڈیا اپ لوڈ کریں"],
        horizontal=True,
    )

    if mode.startswith("Paste"):
        text = st.text_area("اردو متن لکھیں یا پیسٹ کریں", height=140)
        if not st.button("Check / چیک کریں"):
            st.stop()
        if not text or not text.strip():
            st.warning("براہ کرم متن لکھیں / Please enter some text.")
            st.stop()
        if agent is None:
            agent = _make_agent()
        with st.spinner("Checking claim / چیک ہو رہا ہے..."):
            result = agent.run(text)
    else:
        uploaded = st.file_uploader(
            "اپنی فائل اپ لوڈ کریں (jpg, png, mp3, mp4, wav)",
            type=["jpg", "png", "mp3", "mp4", "wav"],
        )
        if uploaded is None:
            st.stop()
        _ensure_models()
        if ingestor is None:
            from ingestion.ingestor import HaqeeqatIngestor

            ingestor = HaqeeqatIngestor()
        if agent is None:
            agent = _make_agent()
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            media_path = tmp.name
        with st.spinner("Processing Media..."):
            report = ingestor.ingest(media_path)
        text = report["combined_text"]
        if not text or not text.strip():
            st.warning("کوئی متن نکالا نہیں جا سکا / No text was extracted.")
            st.stop()
        with st.spinner("Processing Media..."):
            result = agent.run(text)

    with st.expander("استخراج شدہ متن / Extracted text", expanded=True):
        st.code(text)

    if not result.is_checkworthy:
        st.info("کوئی قابلِ تصدیق دعویٰ نہیں / No checkworthy claim found")
        return

    _render_verdict(st, result)
    _render_evidence(st, result)


def _make_agent():
    from verification.verdict_agent import VerdictAgent

    return VerdictAgent()


def _ensure_api_key(st):
    if os.environ.get("GROQ_API_KEY"):
        return
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    if key:
        os.environ["GROQ_API_KEY"] = key


def _ensure_models():
    root = Path(__file__).resolve().parent
    models_dir = root / "models"
    if all((models_dir / name).is_file() for name in _MODEL_FILES):
        return
    subprocess.run(
        [sys.executable, str(root / "tools" / "download_urdu_models.py")],
        check=True,
    )


def _inject_css(st):
    st.markdown(
        """<style>
        .urdu-rtl {
            direction: rtl;
            font-family: "Noto Nastaliq Urdu", "Jameel Noori Nastaleeq", serif;
            font-size: 22px;
            line-height: 1.9;
        }
        .english {
            font-size: 16px;
            margin-top: 4px;
        }
        </style>""",
        unsafe_allow_html=True,
    )


def _bilingual_block(label, urdu, english):
    parts = [f"<p><b>{label}</b></p>"]
    if urdu:
        parts.append(f'<div class="urdu-rtl">{urdu}</div>')
    if english:
        parts.append(f'<div class="english">{english}</div>')
    return "".join(parts)


def _render_verdict(st, result):
    key = result.verdict.value
    label = f"{_URDU_LABELS[key]} ({_ENGLISH_LABELS[key]})"
    message = f"{_ICONS[key]} فیصلہ / Verdict: {label} · Confidence: {result.confidence:.2f}"
    getattr(st, _BOXES[key])(message)
    st.markdown(
        _bilingual_block("دعویٰ / Claim", result.claim_urdu, result.claim_english),
        unsafe_allow_html=True,
    )
    st.markdown(
        _bilingual_block(
            "وجوہات / Reasoning", result.reasoning_urdu, result.reasoning_english
        ),
        unsafe_allow_html=True,
    )


def _render_evidence(st, result):
    sidebar = st.sidebar
    sidebar.header("شواہد / Sources")
    if not result.evidence:
        sidebar.caption("کوئی ذرائع نہیں ملے / No sources found.")
        return
    for item in result.evidence:
        sidebar.markdown(f"- [{item.title}]({item.url})")
        if item.snippet:
            sidebar.caption(item.snippet[:160])


if __name__ == "__main__":
    main()
