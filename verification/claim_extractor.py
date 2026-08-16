import json
import re

from .base import SearchableClaim
from .config import MAX_RETRIES, MODEL_ID, debug_enabled, get_api_key

CLAIM_EXTRACTION_SYSTEM_PROMPT = """\
You are the claim-extraction step of an Urdu fact-checking pipeline.
You receive raw, noisy text: OCR from an image, a speech-to-text transcript, or
both. The text may be written in Urdu script (اردو رسم الخط) OR in Roman Urdu
(Urdu written with English/Latin letters, e.g. "sarkar ne naya qanoon pass kiya").
Decide whether the text contains ONE verifiable factual claim, and if so extract it.

RULES:
1. A claim is check-worthy ONLY if it asserts a verifiable fact about the real
   world: politics, economy, health, religion, sports, crime, viral rumors, etc.
2. IGNORE as not check-worthy: greetings, pleasantries, personal opinions,
   poetry, personal stories or experiences, pure questions, jokes, and any
   content with no factual assertion.
 2b. EXCEPTION (attributed quotes): a statement presented as the words of a
   SPECIFIC named public figure — explicit attribution ("X said", "X نے کہا"),
   a signed quote ("— X", "…! X"), or "X: ..." — IS check-worthy even if
   poetic, lyrical, or humorous, because "did X actually say this?" is a
   verifiable fact. "X" MUST be a proper noun (a specific person's name).
 2c. PROHIBITED: never frame a claim as an attributed quote unless a specific
   person's NAME appears. A speaker that is only a role or generic subject —
   "ترجمان" (spokesman), "حکومت" (government), "وزیرِ اعلیٰ", "the spokesman",
   "officials" — is NOT a named person. For such text, extract the underlying
   factual assertion as an ordinary news claim. Example: "ترجمان نے کہا کہ
   پاکستان نے دفاعی معاہدے میں توسیع کی" → urdu_claim "پاکستان نے دفاعی
   معاہدے میں توسیع کی", NOT "کیا ترجمان نے یہ کہا...؟".
3. If several claims exist, extract only the SINGLE most viral or important one.
4. "urdu_claim": the claim as ONE concise sentence, ALWAYS written in proper
   Urdu script (اردو رسم الخط). If the original is in Roman Urdu, transliterate
   it into proper Urdu script. Preserve the original wording where possible;
   never add your own interpretation.
5. "english_claim": a faithful English translation of urdu_claim, written for
   international web search (mention "Pakistan" if the claim concerns it).
6. "is_checkworthy": true ONLY if a verifiable claim passes rules 1-5 (including 2b).

Respond ONLY with a JSON object, exactly this shape (no prose, no markdown):
{"is_checkworthy": true, "urdu_claim": "...", "english_claim": "..."}"""

_URDU_EXPLICIT_RE = re.compile(
    r"([\u0600-\u06FF][\u0600-\u06FF\s]{2,40}?)\s+"
    r"(?:نے\s+کہا|کہتی\s+ہیں|کہتے\s+ہیں|کہا\s+کہ|کے\s+مطابق|کا\s+قول)"
)
_ENGLISH_EXPLICIT_RE = re.compile(
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\s+"
    r"(?:said|says|stated|told|claimed)\b"
)
_COLON_QUOTE_RE = re.compile(
    r"^([A-Za-z\u0600-\u06FF][\w\u0600-\u06FF\s]{1,40}):\s*[\"\u201c«]",
    re.MULTILINE,
)
_SIGNATURE_RE = re.compile(
    r"(?:—|–|-|!|۔)\s*([\w\u0600-\u06FF][\w\u0600-\u06FF\s]{1,40})\s*$"
)
_FORMAT_MARKS_RE = re.compile(r"[\u061c\u200b-\u200f\u202a-\u202e\u2066-\u2069]")
_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")
_ROLE_ATTRIBUTION_URDU_RE = re.compile(
    r"^کیا\s+(.*?)\s+نے\s+(?:یہ\s+)?کہا(?:\s+کہ\s+)?(.+?)[؟\s]*$",
    re.DOTALL,
)
_ROLE_ATTRIBUTION_ENGLISH_RE = re.compile(
    r"^Did\s+(.*?)\s+say\s+(?:that\s+)?(.+?)\??\s*$", re.DOTALL
)

_ROLE_WORDS = frozenset(
    {
        "ترجمان", "ترجوان", "حکومت", "حکام", "وزیر", "وزراء", "وزیراعظم",
        "صدر", "نگران", "سپیکر", "چیئرمین", "سیکرٹری", "وفد", "سفیر",
        "پولیس", "فوج", "عدالت", "کمپنی", "ادارہ", "تنظیم", "کمیشن", "بورڈ",
        "مشیر", "چیف", "جج", "وکیل", "ماہرین", "ذرائع", "عہدیدار", "افسران",
        "وزارت", "وفاقی", "صوبائی", "ڈپٹی", "گورنر", "کمشنر", "ڈائریکٹر",
        "سرکاری", "پارٹی", "حکمران",
        "the", "government", "official", "officials", "spokesman",
        "spokeswoman", "spokesperson", "ministry", "minister", "president",
        "pm", "army", "police", "court", "delegation", "company", "sources",
        "secretary", "director", "chairman", "commissioner", "chief", "judge",
        "lawyer", "party", "committee", "council", "department", "authorities",
        "agencies", "station", "military", "security",
    }
)


def _normalize_token(token: str) -> str:
    return _DIACRITICS_RE.sub("", token).strip().lower()


_FILLER_WORDS = frozenset(
    {"کہ", "اور", "تو", "پھر", "بھی", "جو", "کیونکہ", "لیکن", "مگر", "یہ", "وہ"}
)


def _strip_fillers(candidate: str) -> str:
    tokens = candidate.split()
    while tokens and _normalize_token(tokens[0]) in _FILLER_WORDS:
        tokens.pop(0)
    return " ".join(tokens)


def _is_role_candidate(candidate: str) -> bool:
    """True if the captured subject is (or contains) a role/generic word."""
    tokens = [_normalize_token(t) for t in candidate.split()]
    return any(token in _ROLE_WORDS for token in tokens)


def detect_attribution(text: str) -> str | None:
    """Return a candidate name if ``text`` exhibits attribution structure.

    Recognises explicit verbal attribution ("X said", "X نے کہا"), a signed
    quote ("— X", "…! X") on the last non-empty line, and a colon-quote
    prefix ("X: "). The candidate is a hint only, never trusted as truth.
    Statements attributed only to a role or generic subject (ترجمان, حکومت,
    وزیرِ اعلیٰ, "the spokesman", ...) are NOT treated as attributed quotes,
    so ordinary news wording is not misread as a viral celebrity claim.
    """
    text = _FORMAT_MARKS_RE.sub("", text or "")
    for pattern in (_URDU_EXPLICIT_RE, _ENGLISH_EXPLICIT_RE):
        for match in pattern.finditer(text or ""):
            candidate = match.group(1).strip()
            if candidate and not _is_role_candidate(candidate):
                return _strip_fillers(candidate)
    colon = _COLON_QUOTE_RE.search(text or "")
    if colon:
        candidate = colon.group(1).strip()
        if candidate and not _is_role_candidate(candidate):
            return _strip_fillers(candidate)
    for line in reversed((text or "").splitlines()):
        if line.strip():
            signature = _SIGNATURE_RE.search(line.strip())
            if signature:
                candidate = signature.group(1).strip()
                if candidate and not _is_role_candidate(candidate):
                    return _strip_fillers(candidate)
            return None
    return None


class ClaimExtractor:
    def __init__(self, groq_client=None, model=MODEL_ID, max_retries=MAX_RETRIES):
        self._client = groq_client
        self.model = model
        self.max_retries = max_retries

    def extract(self, text: str) -> SearchableClaim:
        name = detect_attribution(text)
        messages = [
            {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": self._user_content(text, name)},
        ]
        parsed = self._chat(messages)
        if parsed is None:
            checkworthy = name is not None
            if debug_enabled():
                print(f"DEBUG extractor parse-failed checkworthy={checkworthy}")
            return SearchableClaim(is_checkworthy=checkworthy)
        parsed = self._reframe_role_attribution(parsed)
        claim = SearchableClaim(
            is_checkworthy=parsed["is_checkworthy"] or name is not None,
            urdu_claim=parsed["urdu_claim"],
            english_claim=parsed["english_claim"],
        )
        if debug_enabled():
            print(
                f"DEBUG extractor checkworthy={claim.is_checkworthy} "
                f"urdu={claim.urdu_claim!r} english={claim.english_claim!r}"
            )
        return claim

    def _reframe_role_attribution(self, parsed: dict) -> dict:
        """Turn "Did <role> say that X?" into the underlying factual claim X."""
        urdu_match = _ROLE_ATTRIBUTION_URDU_RE.match(parsed.get("urdu_claim", ""))
        if urdu_match and _is_role_candidate(urdu_match.group(1)):
            content = urdu_match.group(2).strip()
            if len(content.split()) >= 3:
                parsed["urdu_claim"] = content
        english_match = _ROLE_ATTRIBUTION_ENGLISH_RE.match(parsed.get("english_claim", ""))
        if english_match and _is_role_candidate(english_match.group(1)):
            content = english_match.group(2).strip()
            if len(content.split()) >= 3:
                parsed["english_claim"] = content
        return parsed

    def _user_content(self, text: str, name: str | None) -> str:
        content = f"<text>\n{text}\n</text>"
        if name:
            content += (
                f'\n\nHINT: the text appears to present a statement as the words '
                f'of a person (possible name: "{name}"). Extract the claim as an '
                f'attribution question: "کیا <person> نے یہ کہا؟" / '
                f'"Did <person> say this?", preserving the exact wording of the '
                f"statement."
            )
        return content

    def _chat(self, messages):
        client = self._get_client()
        last_content = ""
        for _ in range(self.max_retries):
            call_messages = list(messages)
            if last_content:
                call_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Previous output was invalid: {last_content}\n"
                            "Return valid JSON only, in exactly the specified shape."
                        ),
                    }
                )
            response = client.chat.completions.create(
                model=self.model,
                messages=call_messages,
                temperature=0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            last_content = response.choices[0].message.content or ""
            parsed = self._parse(last_content)
            if parsed is not None:
                return parsed
        return None

    def _parse(self, content: str):
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{.*\}", content or "", re.DOTALL)
            if match is None:
                return None
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        if not isinstance(data, dict):
            return None
        is_checkworthy = data.get("is_checkworthy")
        urdu_claim = data.get("urdu_claim")
        english_claim = data.get("english_claim")
        if not isinstance(is_checkworthy, bool):
            return None
        if not isinstance(urdu_claim, str) or not urdu_claim.strip():
            return None
        if not isinstance(english_claim, str) or not english_claim.strip():
            return None
        return {
            "is_checkworthy": is_checkworthy,
            "urdu_claim": urdu_claim.strip(),
            "english_claim": english_claim.strip(),
        }

    def _get_client(self):
        if self._client is None:
            from groq import Groq

            api_key = get_api_key()
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Create a .env file at the project "
                    "root with GROQ_API_KEY=<your key>."
                )
            self._client = Groq(api_key=api_key)
        return self._client
