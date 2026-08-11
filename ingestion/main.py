"""Command-line entry point for the ingestion module.

Usage:
    python -m ingestion.main <path-to-image-or-audio-or-video>
"""

import argparse
import json
import sys

from ingestion.base import UnsupportedFormatError
from ingestion.ingestor import HaqeeqatIngestor


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="haqeeqat-ingest",
        description="Extract text (speech + on-screen) from an image, audio, or video file.",
    )
    parser.add_argument("file", help="Path to an image, audio, or video file")
    args = parser.parse_args(argv)

    ingestor = HaqeeqatIngestor()
    try:
        report = ingestor.ingest(args.file)
    except UnsupportedFormatError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
