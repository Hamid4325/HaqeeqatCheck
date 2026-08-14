import sys

from .verdict_agent import VerdictAgent


def main(argv=None, ingestor=None, agent=None):
    argv = sys.argv if argv is None else argv
    if len(argv) != 2:
        print(
            "Usage: python -m verification.app <image|audio|video file>",
            file=sys.stderr,
        )
        return 2
    path = argv[1]

    if ingestor is None:
        from ingestion.ingestor import HaqeeqatIngestor

        ingestor = HaqeeqatIngestor()
    if agent is None:
        agent = VerdictAgent()

    report = ingestor.ingest(path)
    text = report["combined_text"]
    if not text.strip():
        print("No text could be extracted from the file.", file=sys.stderr)
        return 1

    result = agent.run(text)
    if not result.is_checkworthy:
        print("کوئی قابلِ تصدیق دعویٰ نہیں")
        return 0

    print(f"دعویٰ: {result.claim_urdu}")
    print(f"فیصلہ: {result.verdict_label_urdu} (confidence {result.confidence:.2f})")
    print(f"وجوہات: {result.reasoning_urdu}")
    print("ذرائع:")
    for item in result.evidence[:3]:
        print(f"  - {item.title} ({item.url})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
