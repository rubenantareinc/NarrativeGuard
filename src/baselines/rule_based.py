from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

NEGATION_CUES = {"not", "never", "no", "none", "cannot", "can't", "won't"}
CONTRADICTION_CUES = {"however", "but", "yet", "although", "despite"}
TEMPORAL_CONFLICTS = {("before", "after"), ("first", "last"), ("always", "never")}


@dataclass
class RuleBasedPrediction:
    label: int
    rationale: str


class RuleBasedBaseline:
    """Simple heuristic baseline for narrative inconsistency detection."""

    def __init__(self, min_negations: int = 1, cue_threshold: int = 2) -> None:
        self.min_negations = min_negations
        self.cue_threshold = cue_threshold

    def predict(self, text: str) -> RuleBasedPrediction:
        tokens = re.findall(r"\b\w+\b", text.lower())
        negations = sum(1 for token in tokens if token in NEGATION_CUES)
        cues = sum(1 for token in tokens if token in CONTRADICTION_CUES)
        temporal_pairs = self._count_temporal_conflicts(tokens)

        score = negations + cues + temporal_pairs
        label = 1 if score >= self.cue_threshold and negations >= self.min_negations else 0
        rationale = (
            f"negations={negations}, cues={cues}, temporal_pairs={temporal_pairs}, score={score}"
        )
        return RuleBasedPrediction(label=label, rationale=rationale)

    def batch_predict(self, texts: Iterable[str]) -> List[RuleBasedPrediction]:
        return [self.predict(text) for text in texts]

    def _count_temporal_conflicts(self, tokens: List[str]) -> int:
        count = 0
        token_set = set(tokens)
        for first, second in TEMPORAL_CONFLICTS:
            if first in token_set and second in token_set:
                count += 1
        return count
