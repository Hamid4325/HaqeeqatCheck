"""Download UTRNet + YOLOv8-UrduDoc weights into the gitignored models/ dir.

Usage:
    python tools/download_urdu_models.py [--force]
"""

import argparse
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
BASE_URL = "https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet/resolve/main"
FILES = {
    "best_norm_ED.pth": 41 * 1024 * 1024,
    "yolov8m_UrduDoc.pt": int(49.7 * 1024 * 1024),
}


def download(name, expected_bytes, force):
    dest = os.path.join(MODELS_DIR, name)
    if os.path.isfile(dest) and os.path.getsize(dest) >= expected_bytes * 0.95 and not force:
        print("skip  %s (already present, %.1f MB)" % (name, os.path.getsize(dest) / 1048576))
        return
    url = "%s/%s" % (BASE_URL, name)
    print("fetch %s ..." % url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        total = 0
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
            if total % (2 * 1048576) < 65536:
                print("  %.1f MB ..." % (total / 1048576))
    print("saved %s (%.1f MB)" % (name, os.path.getsize(dest) / 1048576))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, size in FILES.items():
        download(name, size, args.force)
    print("Done. Models are in %s" % MODELS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
