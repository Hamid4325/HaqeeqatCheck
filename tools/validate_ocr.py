"""OCR validation harness: PaddleOCR (baseline) vs EasyOCR on test images.

Usage:
    python tools/validate_ocr.py <image> [<image>...]

For each image it runs PaddleOCR (Urdu) as the baseline and EasyOCR (Urdu,
CPU-only) as the candidate, prints a side-by-side comparison as repr() so
Urdu text renders safely, and appends the same results to
`.superpowers/sdd/ocr-validation.md`.

PaddleOCR is executed in a subprocess because it can crash natively (access
violation) on large images; that cannot be caught with try/except in-process.
Isolating it lets the run continue and still record the EasyOCR result.
"""

import datetime
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_FIRST = "شکریہ پاکستان"

MD_PATH = os.path.join(ROOT, ".superpowers", "sdd", "ocr-validation.md")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
del _stream

PADDLE_WORKER = r"""
import sys

try:
    from ingestion.paddle_ocr_engine import PaddleOCREngine
    engine = PaddleOCREngine(lang="ur")
    text = engine.extract_text(sys.argv[1])
    print("RESULT:" + repr(text))
except Exception as exc:
    print("ERROR: %s: %s" % (type(exc).__name__, exc))
"""


def paddle_baseline(image_path):
    """Return the PaddleOCR result (repr string) or an 'ERROR: ...' string.

    Runs in a subprocess so a native access violation cannot kill this
    process. The worker prints either 'RESULT:<repr>' or 'ERROR: ...'.
    """
    cmd = [sys.executable, "-c", PADDLE_WORKER, image_path]
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return "ERROR: TimeoutExpired: PaddleOCR did not finish within 600s"
    for line in (proc.stdout or "").splitlines():
        if line.startswith("RESULT:"):
            return line[len("RESULT:"):]
        if line.startswith("ERROR:"):
            return line[len("ERROR:"):]
    detail = (proc.stderr or "").strip().splitlines()
    detail = detail[-1].strip() if detail else ""
    return "ERROR: subprocess exit %d: %s" % (proc.returncode, detail[:300])


def easyocr_text(reader, image_path):
    """Run EasyOCR and join all detected text lines with '\\n'."""
    results = reader.readtext(image_path, detail=1, paragraph=False)
    lines = []
    for item in results:
        try:
            text = item[1]
        except (TypeError, IndexError, KeyError):
            continue
        if isinstance(text, str) and text.strip():
            lines.append(text.strip())
    return "\n".join(lines)


def main(argv):
    if len(argv) < 2:
        print("usage: python tools/validate_ocr.py <image> [<image>...]",
              file=sys.stderr)
        return 2
    images = [os.path.abspath(path) for path in argv[1:]]

    exit_code = 0
    reader = None
    try:
        from easyocr import Reader
        reader = Reader(["ur"], gpu=False)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print("ERROR: easyocr unavailable: %s: %s" % (type(exc).__name__, exc),
              file=sys.stderr)
        exit_code = 1

    run_date = datetime.date.today().isoformat()
    os.makedirs(os.path.dirname(MD_PATH), exist_ok=True)

    md_lines = []

    def record(*texts):
        for text in texts:
            print(text)
            md_lines.append(text)

    for index, image_path in enumerate(images):
        record("")
        record("## Run date: %s - image: %s" % (run_date, os.path.basename(image_path)))
        if index == 0:
            record("Expected text (first image): %r" % EXPECTED_FIRST)

        record("engine: PaddleOCR (baseline)")
        record("detected: %r" % paddle_baseline(image_path))

        record("engine: EasyOCR")
        if reader is None:
            record("detected: %r" % "ERROR: easyocr unavailable")
            continue
        try:
            text = easyocr_text(reader, image_path)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            text = "ERROR: %s: %s" % (type(exc).__name__, exc)
            exit_code = 1
        record("detected: %r" % text)

    with open(MD_PATH, "a", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines) + "\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
