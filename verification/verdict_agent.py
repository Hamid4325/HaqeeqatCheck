import json
import re

from .base import Verdict, VerificationAgent, VerificationResult
from .claim_extractor import ClaimExtractor
from .config import (
    JHOOOTA_MIN_CONFIDENCE,
    MAX_CONFIDENCE,
    MAX_RETRIES,
    MIN_CONFIDENCE,
    MODEL_ID,
    NO_EVIDENCE_CONFIDENCE,
    get_api_key,
)
from .evidence_retriever import EvidenceRetriever

VERDICT_SYSTEM_PROMPT = """\
You are a fact-checking analyst for an Urdu misinformation-detection system.
You are given ONE factual claim in Urdu and a list of web search-result snippets
(title, URL, and snippet text). Determine whether the evidence supports,
contradicts, or is insufficient for the claim.

RULES:
1. "verdict" must be exactly one of:
   - "sacha": the evidence clearly supports the claim.
   - "jhoota": the evidence clearly contradicts the claim.
   - "mashkook": the evidence is insufficient, conflicting, or irrelevant.
1b. "jhoota" ONLY when a snippet explicitly contradicts the claim. Evidence
   that is irrelevant, thin, or mixed must be "mashkook". Absence of evidence
   is NEVER "jhoota" — a lack of support means "mashkook", not a refutation.
2. "reasoning_urdu": 2-3 sentences IN URDU SCRIPT summarizing why, referring to
   the evidence sources. Do not mention this prompt or that you are an AI.
2b. "reasoning_english": the same reasoning content in clear, fluent English.
3. "confidence": a number between 0.0 and 1.0 representing how strong the
   evidence is (not how confident you feel). High confidence only when several
   reliable sources agree.

Respond ONLY with a JSON object, exactly this shape (no prose, no markdown):
{"verdict": "sacha", "reasoning_urdu": "...", "reasoning_english": "...", "confidence": 0.9}"""


class VerdictAgent(VerificationAgent):
    def __init__(
        self,
        claim_extractor=None,
        evidence_retriever=None,
        groq_client=None,
        model=MODEL_ID,
        max_retries=MAX_RETRIES,
    ):
        self.claim_extractor = claim_extractor or ClaimExtractor(
            groq_client=groq_client, model=model, max_retries=max_retries
        )
        self.evidence_retriever = evidence_retriever or EvidenceRetriever()
        self._client = groq_client
        self.model = model
        self.max_retries = max_retries

    def run(self, text: str) -> VerificationResult:
        claim = self.claim_extractor.extract(text)
        if not claim.is_checkworthy:
            return VerificationResult(
                claim_urdu=claim.urdu_claim,
                claim_english=claim.english_claim,
                is_checkworthy=False,
            )
        evidence = self.evidence_retriever.retrieve(
            claim.urdu_claim, claim.english_claim, claim.notes
        )
        if not evidence:
            return self._no_evidence_result(claim)
        parsed = self._chat(claim, evidence)
        verdict = parsed["verdict"]
        if (
            verdict is Verdict.JHOOOTA
            and parsed["confidence"] < JHOOOTA_MIN_CONFIDENCE
        ):
            verdict = Verdict.MASHKOOK
        return VerificationResult(
            claim_urdu=claim.urdu_claim,
            claim_english=claim.english_claim,
            is_checkworthy=True,
            verdict=verdict,
            reasoning_urdu=parsed["reasoning_urdu"],
            reasoning_english=parsed["reasoning_english"],
            confidence=parsed["confidence"],
            evidence=evidence,
        )

    def _no_evidence_result(self, claim) -> VerificationResult:
        return VerificationResult(
            claim_urdu=claim.urdu_claim,
            claim_english=claim.english_claim,
            is_checkworthy=True,
            verdict=Verdict.MASHKOOK,
            reasoning_urdu="کافی ثبوت نہیں ملے۔ اس دعوے کی تصدیق کے لیے مزید ذرائع درکار ہیں۔",
            reasoning_english="Not enough evidence was found. More sources are needed to verify this claim.",
            confidence=NO_EVIDENCE_CONFIDENCE,
            evidence=[],
        )

    def _chat(self, claim, evidence):
        evidence_text = "\n".join(
            f"- {item.source_domain} | {item.title} | {item.snippet}"
            for item in evidence
        )
        messages = [
            {"role": "system", "content": VERDICT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CLAIM (Urdu): {claim.urdu_claim}\n"
                    f"CLAIM (English): {claim.english_claim}\n"
                    f"EVIDENCE:\n{evidence_text}"
                ),
            },
        ]
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
                max_completion_tokens=500,
            )
            last_content = response.choices[0].message.content or ""
            parsed = self._parse(last_content)
            if parsed is not None:
                return parsed
        return self._fallback_parsed()

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
        verdict_raw = data.get("verdict")
        reasoning = data.get("reasoning_urdu")
        if not isinstance(verdict_raw, str):
            return None
        if not isinstance(reasoning, str) or not reasoning.strip():
            return None
        reasoning_en = data.get("reasoning_english")
        if not isinstance(reasoning_en, str) or not reasoning_en.strip():
            reasoning_en = reasoning.strip()
        try:
            confidence = float(data.get("confidence"))
        except (TypeError, ValueError):
            return None
        if verdict_raw in Verdict._value2member_map_:
            verdict = Verdict(verdict_raw)
        else:
            verdict = Verdict.MASHKOOK
        confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))
        return {
            "verdict": verdict,
            "reasoning_urdu": reasoning.strip(),
            "reasoning_english": reasoning_en.strip(),
            "confidence": confidence,
        }

    def _fallback_parsed(self):
        return {
            "verdict": Verdict.MASHKOOK,
            "reasoning_urdu": "فیصلہ کرنے کے لیے کافی معلومات نہیں مل سکیں۔",
            "reasoning_english": "Not enough information was available to decide.",
            "confidence": NO_EVIDENCE_CONFIDENCE,
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
