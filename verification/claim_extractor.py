import json
import re

from .base import SearchableClaim
from .config import MAX_RETRIES, MODEL_ID, get_api_key

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
3. If several claims exist, extract only the SINGLE most viral or important one.
4. "urdu_claim": the claim as ONE concise sentence, ALWAYS written in proper
   Urdu script (اردو رسم الخط). If the original is in Roman Urdu, transliterate
   it into proper Urdu script. Preserve the original wording where possible;
   never add your own interpretation.
5. "english_claim": a faithful English translation of urdu_claim, written for
   international web search (mention "Pakistan" if the claim concerns it).
6. "is_checkworthy": true ONLY if a verifiable claim passes rules 1-5.

Respond ONLY with a JSON object, exactly this shape (no prose, no markdown):
{"is_checkworthy": true, "urdu_claim": "...", "english_claim": "..."}"""


class ClaimExtractor:
    def __init__(self, groq_client=None, model=MODEL_ID, max_retries=MAX_RETRIES):
        self._client = groq_client
        self.model = model
        self.max_retries = max_retries

    def extract(self, text: str) -> SearchableClaim:
        messages = [
            {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"<text>\n{text}\n</text>"},
        ]
        parsed = self._chat(messages)
        if parsed is None:
            return SearchableClaim(is_checkworthy=False)
        return SearchableClaim(
            is_checkworthy=parsed["is_checkworthy"],
            urdu_claim=parsed["urdu_claim"],
            english_claim=parsed["english_claim"],
        )

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
                temperature=0.2,
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
