import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class Verdict(str, enum.Enum):
    SACHA = "sacha"
    JHOOOTA = "jhoota"
    MASHKOOK = "mashkook"


URDU_LABELS = {
    "sacha": "سچا",
    "jhoota": "جھوٹا",
    "mashkook": "مشکوک",
}


@dataclass
class EvidenceItem:
    title: str
    url: str
    snippet: str
    source_domain: str


@dataclass
class SearchableClaim:
    is_checkworthy: bool = False
    urdu_claim: str = ""
    english_claim: str = ""


@dataclass
class VerificationResult:
    claim_urdu: str = ""
    claim_english: str = ""
    is_checkworthy: bool = False
    verdict: Verdict | None = None
    reasoning_urdu: str = ""
    confidence: float = 0.0
    evidence: list[EvidenceItem] = field(default_factory=list)

    @property
    def verdict_label_urdu(self) -> str:
        if self.verdict is None:
            return ""
        return URDU_LABELS[self.verdict.value]


class VerificationAgent(ABC):
    @abstractmethod
    def run(self, text: str) -> VerificationResult:
        ...
