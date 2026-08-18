"""Haqeeqat Check — Gradio interface for Hugging Face Spaces.

Run locally:  python app.py
"""

import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure GROQ_API_KEY is loaded from Space secrets before any module import
# ---------------------------------------------------------------------------
if not os.environ.get("GROQ_API_KEY"):
    pass  # will be read at runtime by verification/config.py

import gradio as gr

try:
    import spaces
    _HAS_SPACES = True
except ImportError:
    _HAS_SPACES = False

# ---------------------------------------------------------------------------
# Model bootstrap — runs once at import time
# ---------------------------------------------------------------------------
_MODEL_FILES = ["best_norm_ED.pth", "yolov8m_UrduDoc.pt"]


def _ensure_models():
    """Download OCR + Whisper models if not already present."""
    root = Path(__file__).resolve().parent
    models_dir = root / "models"
    if os.environ.get("SPACE_ID"):
        models_dir = Path("/home/user/app/models")
    if all((models_dir / name).is_file() for name in _MODEL_FILES):
        return
    subprocess.run(
        [sys.executable, str(root / "download_models.py")],
        check=True,
    )


_ensure_models()

# ---------------------------------------------------------------------------
# Lazy singletons — heavy imports deferred until first request
# ---------------------------------------------------------------------------
_ingestor = None
_agent = None


def _get_ingestor():
    global _ingestor
    if _ingestor is None:
        from ingestion.ingestor import HaqeeqatIngestor
        _ingestor = HaqeeqatIngestor()
    return _ingestor


def _get_agent():
    global _agent
    if _agent is None:
        from verification.verdict_agent import VerdictAgent
        _agent = VerdictAgent()
    return _agent


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
URDU_LABELS = {"sacha": "سچا", "jhoota": "جھوٹا", "mashkook": "مشکوک"}
ENGLISH_LABELS = {"sacha": "True", "jhoota": "False", "mashkook": "Unverified"}
VERDICT_ICONS = {"sacha": "✔", "jhoota": "✗", "mashkook": "?"}


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

if _HAS_SPACES:
    @spaces.GPU
    def _gpu_startup():
        """Dummy function so Gradio 6.x detects a GPU-capable handler at startup."""
        pass


def _resolve_path(file_data) -> str | None:
    """Extract a filesystem path from a Gradio FileData object or plain string."""
    if isinstance(file_data, str):
        return file_data
    # Gradio 6.x FileData: object with .path or dict-like access
    if hasattr(file_data, "path"):
        return file_data.path
    if isinstance(file_data, dict):
        return file_data.get("path")
    return None


def _process_media_inner(file_data) -> tuple[str, str]:
    """Ingest a media file, verify claims, return (verdict_box, reasoning_box)."""
    from verification.debug_trace import get_trace

    if file_data is None:
        return "کوئی فائل منتخب نہیں / No file selected.", ""

    # Gradio 6.x passes a FileData object; extract the path string
    file_path = _resolve_path(file_data)
    if not file_path:
        return "کوئی فائل منتخب نہیں / No file selected.", ""

    ingestor = _get_ingestor()
    agent = _get_agent()

    report = ingestor.ingest(file_path)
    text = report.get("combined_text", "")
    if not text or not text.strip():
        return "کوئی متن نکالا نہیں جا سکا / No text was extracted from the file.", ""

    if report.get("metadata", {}).get("ocr_garbled"):
        return (
            "تصحیح OCR ناکام رہی / OCR failed to read this image properly.\n"
            "براہ کرم واضح تصویر اپ لوڈ کریں / Please upload a clearer image."
        ), f"استخراج شدہ متن:\n{text[:300]}"

    result = agent.run(text)
    trace_lines = get_trace()

    if not result.is_checkworthy:
        debug_info = "\n".join(trace_lines) if trace_lines else ""
        return "کوئی قابلِ تصدیق دعویٰ نہیں / No checkworthy claim found.", (
            f"استخراج شدہ متن:\n{text[:500]}\n\n--- DEBUG TRACE ---\n{debug_info}"
        )

    verdict_box = _format_verdict(result)
    reasoning_box = _format_reasoning(result)
    if trace_lines:
        reasoning_box += "\n\n--- DEBUG TRACE ---\n" + "\n".join(trace_lines)
    return verdict_box, reasoning_box


def _process_text(text: str) -> tuple[str, str]:
    """Verify a pasted text claim, return (verdict_box, reasoning_box)."""
    from verification.debug_trace import get_trace

    if not text or not text.strip():
        return "براہ کرم متن لکھیں / Please enter some text.", ""

    agent = _get_agent()
    result = agent.run(text)
    trace_lines = get_trace()

    if not result.is_checkworthy:
        debug_info = "\n".join(trace_lines) if trace_lines else ""
        return "کوئی قابلِ تصدیق دعویٰ نہیں / No checkworthy claim found.", (
            f"--- DEBUG TRACE ---\n{debug_info}"
        )

    verdict_box = _format_verdict(result)
    reasoning_box = _format_reasoning(result)
    if trace_lines:
        reasoning_box += "\n\n--- DEBUG TRACE ---\n" + "\n".join(trace_lines)
    return verdict_box, reasoning_box


# Wrap _process_media_inner with @spaces.GPU on HF so UTRNet gets a GPU.
if _HAS_SPACES:
    @spaces.GPU
    def _process_media(file_data) -> tuple[str, str]:
        return _process_media_inner(file_data)
else:
    _process_media = _process_media_inner


def _format_verdict(result) -> str:
    key = result.verdict.value
    icon = VERDICT_ICONS[key]
    urdu_label = URDU_LABELS[key]
    eng_label = ENGLISH_LABELS[key]
    lines = [
        f"{icon}  فیصلہ / Verdict:  {urdu_label} ({eng_label})",
        f" Confidence: {result.confidence:.0%}",
        "",
        f"دعویٰ / Claim:",
        f"  {result.claim_urdu}",
        f"  {result.claim_english}",
    ]
    return "\n".join(lines)


def _format_reasoning(result) -> str:
    parts = [
        "وجوہات / Reasoning:",
        "",
        result.reasoning_urdu,
        "",
        result.reasoning_english,
    ]
    if result.evidence:
        parts.append("")
        parts.append("شواہد / Sources:")
        for item in result.evidence:
            parts.append(f"  [{item.source_domain}] {item.title}")
            parts.append(f"  {item.url}")
            if item.snippet:
                parts.append(f"  {item.snippet[:200]}")
            parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Haqeeqat Check — Urdu Misinformation Detector",
    ) as demo:
        gr.Markdown(
            "# Uraan Techathon 2.0\n"
            "# Haqeeqat Check: Urdu Misinformation Detector\n"
            "### حقیقت چیک: اردو غلط معلومات کی جانچ پڑتال"
        )

        with gr.Tabs():
            with gr.Tab("Image"):
                img_input = gr.Image(label="تصویر اپ لوڈ کریں / Upload Image", type="filepath")
                img_btn = gr.Button("Check / چیک کریں", variant="primary")

            with gr.Tab("Audio"):
                aud_input = gr.Audio(label="آڈیو اپ لوڈ کریں / Upload Audio", type="filepath")
                aud_btn = gr.Button("Check / چیک کریں", variant="primary")

            with gr.Tab("Video"):
                vid_input = gr.Video(label="ویڈیو اپ لوڈ کریں / Upload Video")
                vid_btn = gr.Button("Check / چیک کریں", variant="primary")

            with gr.Tab("Paste Text"):
                txt_input = gr.Textbox(
                    label="اردو متن لکھیں یا پیسٹ کریں / Enter or paste Urdu text",
                    lines=5,
                )
                txt_btn = gr.Button("Check / چیک کریں", variant="primary")

        gr.Markdown("---")
        verdict_output = gr.Textbox(
            label="فیصلہ / Verdict",
            lines=8,
            interactive=False,
        )
        reasoning_output = gr.Textbox(
            label="وجوہات و شواہد / Reasoning & Sources",
            lines=14,
            interactive=False,
        )

        img_btn.click(fn=_process_media, inputs=img_input, outputs=[verdict_output, reasoning_output])
        aud_btn.click(fn=_process_media, inputs=aud_input, outputs=[verdict_output, reasoning_output])
        vid_btn.click(fn=_process_media, inputs=vid_input, outputs=[verdict_output, reasoning_output])
        txt_btn.click(fn=_process_text, inputs=txt_input, outputs=[verdict_output, reasoning_output])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(theme=gr.themes.Soft())
