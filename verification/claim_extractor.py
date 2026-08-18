import json
import re

from .base import SearchableClaim
from .config import MAX_RETRIES, MODEL_ID, debug_enabled, get_api_key
from .debug_trace import trace

CLAIM_EXTRACTION_SYSTEM_PROMPT = """\
You are the claim-extraction step of an Urdu fact-checking pipeline.
You receive raw, noisy text: OCR from an image, a speech-to-text transcript, or
both. The text may be written in Urdu script (اردو رسم الخط) OR in Roman Urdu
(Urdu written with English/Latin letters, e.g. "sarkar ne naya qanoon pass kiya").
Decide whether the text contains ONE verifiable factual claim, and if so extract it.

RULES:
1. A claim is check-worthy ONLY if it asserts a verifiable fact about the real
   world: politics, economy, health, religion, sports, crime, viral rumors, etc.
   INCLUDES: news reports, press conferences, government announcements, TV news
   transcripts, and official statements. If the text mentions a country name
   (e.g. Pakistan, Saudi Arabia, Turkey, Iran, America) AND describes an action
   or event (agreement, death, accident, policy, price change, court ruling,
   military operation, etc.), that is a check-worthy claim.
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
        "media", "میڈیا", "می میڈیا", "express", "dawn", "geo", "ary",
        "the", "government", "official", "officials", "spokesman",
        "spokeswoman", "spokesperson", "ministry", "minister", "president",
        "pm", "army", "police", "court", "delegation", "company", "sources",
        "secretary", "director", "chairman", "commissioner", "chief", "judge",
        "lawyer", "party", "committee", "council", "department", "authorities",
        "agencies", "station", "military", "security",
    }
)


def _clean_text(text: str) -> str:
    """Strip Unicode formatting marks, diacritics, and excessive whitespace."""
    text = _FORMAT_MARKS_RE.sub("", text or "")
    text = _DIACRITICS_RE.sub("", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = " ".join(lines)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


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
        trace(f"[Extractor] Input text length: {len(text)} chars")
        name = detect_attribution(text)
        trace(f"[Extractor] Attribution detected: {name!r}")
        messages = [
            {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": self._user_content(text, name)},
        ]
        trace(f"[Extractor] Sending to LLM (model={self.model})...")
        parsed = self._chat(messages)
        if parsed is None:
            checkworthy = name is not None
            trace(f"[Extractor] LLM returned None (parse failed). checkworthy={checkworthy}")
            if not checkworthy:
                is_news = self._looks_like_news(text)
                trace(f"[Extractor] LLM parse failed, _looks_like_news={is_news}")
                if is_news:
                    trace(f"[Extractor] LLM parse failed but text looks like news, retrying...")
                    claim = self._retry_as_news_claim(text, name)
                    trace(f"[Extractor] Retry result: checkworthy={claim.is_checkworthy} urdu={claim.urdu_claim!r}")
                    return claim
                else:
                    claim = self._heuristic_extract(text)
                    if claim.is_checkworthy:
                        trace(f"[Extractor] Heuristic fallback: checkworthy=True urdu={claim.urdu_claim!r}")
                        return claim
            return SearchableClaim(is_checkworthy=checkworthy)
        trace(f"[Extractor] LLM returned: checkworthy={parsed['is_checkworthy']} urdu={parsed['urdu_claim']!r} english={parsed['english_claim']!r}")
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
        if not claim.is_checkworthy:
            trace(f"[Extractor] LLM said not checkworthy. Checking _looks_like_news...")
            is_news = self._looks_like_news(text)
            trace(f"[Extractor] _looks_like_news={is_news}")
            if is_news:
                trace(f"[Extractor] Retrying as news claim...")
                claim = self._retry_as_news_claim(text, name)
                trace(f"[Extractor] Retry result: checkworthy={claim.is_checkworthy} urdu={claim.urdu_claim!r}")
        elif not claim.urdu_claim.strip():
            trace(f"[Extractor] Attribution forced checkworthy but claim is empty. Retrying...")
            if self._looks_like_news(text):
                claim = self._retry_as_news_claim(text, name)
                trace(f"[Extractor] Retry result: checkworthy={claim.is_checkworthy} urdu={claim.urdu_claim!r}")
            else:
                claim = self._heuristic_extract(text)
                trace(f"[Extractor] Heuristic result: checkworthy={claim.is_checkworthy} urdu={claim.urdu_claim!r}")
        else:
            trace(f"[Extractor] Claim IS checkworthy, proceeding.")
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

    _NEWS_COUNTRY_RE = re.compile(
        r"(?:پاکستان|پاکسان|ایران|امریکہ|ترکی|ترکے|سعودی|چین|ہندوستان|افغانستان|عراق|"
        r"Pakistan|Iran|America|Turkey|Saudi|China|India|Afghanistan|Iraq)"
    )
    _NEWS_ACTION_RE = re.compile(
        r"(?:defense|defences|security|agreement|treaty|moaida|"
        r"attack|explosion|death|killed|arrest|court|ruling|price|inflation|"
        r"\u0641\u0648\u062c|\u062f\u0641\u0627\u0626\u06cc|\u0627\u0645\u0646|\u062a\u0639\u0627\u0648\u0646"
        r"|\u062d\u0645\u0644\u06d0|\u062f\u06be\u0645\u0627\u06a9\u06d0|\u0645\u0648\u062a"
        r"|\u06af\u0631\u0641\u062a\u0627\u0631|\u0639\u062f\u0627\u0644\u062a|\u0641\u06cc\u0635\u0644\u06d0"
        r"|\u0642\u06cc\u0645\u062a|\u0645\u0646\u0647\u0627\u0626\u06cc"
        r"|\u0645\u0648\u062d\u062f\u06d0|\u0645\u0639\u0627\u0647\u062f\u06d0"
        r"|\u0627\u0632\u0627\u0645|\u062a\u062d\u0642\u06cc\u0642|\u0627\u0639\u0644\u0627\u0646"
        r"|\u062a\u0648\u0633\u06cc\u0639|\u0627\u0633\u062a\u062b\u0646\u0627\u0626\u06d0|\u0641\u0631\u0627\u0647\u0645"
        r"|barish|flood|accident|crash|election|intekhab|budget|baajpata)",
        re.IGNORECASE,
    )

    def _looks_like_news(self, text: str) -> bool:
        cleaned = _clean_text(text)
        if len(cleaned) < 80:
            return False
        has_country = bool(self._NEWS_COUNTRY_RE.search(cleaned))
        has_action = bool(self._NEWS_ACTION_RE.search(cleaned))
        return has_country and has_action

    _NEWS_RETRY_PROMPT = """\
The text below is from a news broadcast or press release. It contains a factual \
claim about a real-world event or policy. Extract the single most important \
verifiable claim and mark it as checkworthy.

Respond ONLY with a JSON object:
{"is_checkworthy": true, "urdu_claim": "...", "english_claim": "..."}"""

    _HEURISTIC_SENTENCE_RE = re.compile(
        r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]"
        r"[^\.\?\u06d4]{10,200}[\.\?\u06d4]?"
    )

    _HEURISTIC_KEYWORDS_RE = re.compile(
        r"(?:\u0646\u06d0\u0602\u0646\u06d0|\u06a9\u06d0\u0602\u0627\u0602\u06d0|\u06af\u0630\u0627\u0634\u062a\u06d0"
        r"|\u0641\u06cc\u0635\u0644\u06d0 \u06a9\u06cc\u0627\u06d0|\u0627\u0639\u0644\u0627\u0646 \u06a9\u0631\u062f\u06d0"
        r"|\u062a\u062d\u0642\u06cc\u0642 \u06a9\u06cc\u0627\u06d0|\u0645\u0646\u0639\u06cc \u06a9\u06cc\u0627\u06d0"
        r"|\u062e\u0627\u0631\u062c\u06d0 \u06a9\u06cc\u0627\u06d0|\u0645\u0648\u062f \u062f\u06cc\u0627|\u0645\u0648\u062d\u062f\u06d0"
        r"|\u062a\u0648\u0633\u06cc\u0639 \u06a9\u06cc\u0627\u06d0|\u0627\u0634\u0627\u0631\u06d0 \u062f\u06cc\u0627"
        r"|\u0627\u0633\u062a\u062b\u0646\u06d0\u06cc \u062f\u06cc\u0627|\u062a\u0631\u062a\u06cc\u0628 \u06a9\u06cc\u0627\u06d0"
        r"|\u06a9\u06cc\u0627\u06d0|\u062f\u0639\u0648\u06cc \u06a9\u06cc\u0627\u06d0|\u0627\u0632\u0627\u0645 \u067e\u0631"
        r"|\u0642\u0627\u0631\u062c\u06d0 \u06a9\u06cc\u0627\u06d0|\u062f\u0648\u0631\u06d0 \u06a9\u06cc\u0627\u06d0"
        r"|\u0645\u0637\u0627\u0644\u0628\u06d0 \u06a9\u06cc\u0627\u06d0|\u0646\u06d0\u0634\u0633\u062a\u06d0 \u06a9\u06cc\u0627\u06d0"
        r"|said|announced|declared|signed|launched|banned|arrested|killed|died|ruled|ordered)",
        re.IGNORECASE,
    )

    def _heuristic_extract(self, text: str) -> SearchableClaim:
        """Extract a claim using regex heuristics when the LLM fails."""
        cleaned = _clean_text(text)
        sentences = self._HEURISTIC_SENTENCE_RE.findall(cleaned)
        trace(f"[Heuristic] Found {len(sentences)} candidate sentences")
        best = None
        best_score = 0
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 15:
                continue
            score = 0
            if self._NEWS_COUNTRY_RE.search(sent):
                score += 2
            if self._HEURISTIC_KEYWORDS_RE.search(sent):
                score += 1
            if self._NEWS_ACTION_RE.search(sent):
                score += 1
            if score > best_score:
                best_score = score
                best = sent
        trace(f"[Heuristic] Best score: {best_score}, best: {best!r}")
        if best is None or best_score < 2:
            return SearchableClaim(is_checkworthy=False)
        claim = best.rstrip(".\u06d4?\u2026 ")
        return SearchableClaim(
            is_checkworthy=True,
            urdu_claim=claim,
            english_claim=claim,
        )

    def _retry_as_news_claim(self, text: str, name: str | None) -> SearchableClaim:
        cleaned = _clean_text(text)
        trace(f"[Retry] Cleaned text length: {len(cleaned)} chars")
        messages = [
            {"role": "system", "content": self._NEWS_RETRY_PROMPT},
            {"role": "user", "content": f"<text>\n{cleaned}\n</text>"},
        ]
        trace(f"[Retry] Sending retry to LLM...")
        parsed = self._chat(messages)
        if parsed is not None and parsed.get("is_checkworthy"):
            trace(f"[Retry] LLM returned checkworthy claim: {parsed['urdu_claim']!r}")
            parsed = self._reframe_role_attribution(parsed)
            return SearchableClaim(
                is_checkworthy=True,
                urdu_claim=parsed["urdu_claim"],
                english_claim=parsed["english_claim"],
            )
        trace(f"[Retry] LLM also failed. Falling back to heuristic extraction.")
        return self._heuristic_extract(text)

    def _user_content(self, text: str, name: str | None) -> str:
        cleaned = _clean_text(text)
        content = f"<text>\n{cleaned}\n</text>"
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
                max_completion_tokens=300,
            )
            last_content = response.choices[0].message.content or ""
            parsed = self._parse(last_content)
            if parsed is not None:
                return parsed
        return None

    def _parse(self, content: str):
        try:
            cleaned = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()
            data = json.loads(cleaned)
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
