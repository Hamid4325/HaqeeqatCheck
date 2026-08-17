import re


_ScamSignal = re.IGNORECASE

_SCAM_KEYWORDS_RE = re.compile(
    r"(?:bisp|ehsaas|ehsan|ایڈی|eidi|identifi|شناخت|پہچان|"
    r"کا?رڈ|card|gover|حکومت|سرکار|وزیر|sarkar|"
    r"8171|786|پنج\s*هزار|5000|ایک\s*سو\s*ایک|"
    r"پکی\s*خبر|verified|پکی|سچی|سمجھو|سمجھ|"
    r"فون|phone|moj| mobil|رقم|amount|rb?sid|رسید)",
    re.IGNORECASE,
)


def has_scam_signals(text: str) -> bool:
    """Return True if *text* contains known scam-related keywords or numbers."""
    if not text:
        return False
    return bool(_SCAM_KEYWORDS_RE.search(text))

_PHONETIC_DIGITS_URDU = {
    "ایک": "1", "اک": "1",
    "دو": "2",
    "تین": "3", "tin": "3",
    "چار": "4", "char": "4",
    "پانچ": "5",
    "چھ": "6", "che": "6",
    "سات": "7", "sat": "7",
    "آٹھ": "8", "ath": "8",
    "نو": "9", "nau": "9",
    "دس": "10", "das": "10",
}

_PHONETIC_RE = re.compile(
    r"(?:ایک|اک|دو|تین|tin|چار|char|پانچ|چھ|che|سات|sat|آٹھ|ath|نو|nau|دس|das)"
    r"(?:\s+(?:و|aur)\s+(?:ایک|اک|دو|تین|tin|چار|char|پانچ|چھ|che|سات|sat|آٹھ|ath|نو|nau|دس|das))+",
    re.IGNORECASE,
)


def extract_numbers(text: str) -> list[str]:
    """Extract numeric tokens from *text*.

    Finds bare digits (e.g. "8171", "786") and also converts phonetic Urdu
    numbers written in script (e.g. "ایٹ ون سیون ون") into digit strings
    when possible.  Returns a list of digit strings.
    """
    if not text:
        return []

    digits: list[str] = re.findall(r"\b\d{4,5}\b", text)

    for m in _PHONETIC_RE.finditer(text):
        raw = m.group(0)
        built = ""
        for token in raw.split():
            if token in ("و", "aur"):
                continue
            built += _PHONETIC_DIGITS_URDU.get(token, "")
        if built and built not in digits:
            digits.append(built)

    return digits


def number_note_for_claim(numbers: list[str]) -> str | None:
    """Return a SearchableClaim note if *numbers* contains 4- or 5-digit
    values that may be tied to known scams or government programs."""
    if not numbers:
        return None
    notable = [n for n in numbers if len(n) in (4, 5)]
    if not notable:
        return None
    joined = ", ".join(notable)
    return (
        f"Numbers found in transcript: {joined}. "
        "Check if any of these numbers are associated with a known scam "
        "or government program."
    )
