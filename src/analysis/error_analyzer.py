from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

ERROR_TYPES = {
    "false_positives": {
        "creative_ambiguity": "System flagged intentional ambiguity as error",
        "domain_specific": "Missed domain-specific valid constructions",
        "context_needed": "Required broader context than available",
    },
    "false_negatives": {
        "subtle_contradiction": "Missed logical but non-obvious contradictions",
        "long_range_dependency": "Failed to track info across long documents",
        "implicit_conflict": "Didn't infer implicit contradictions",
    },
    "edge_cases": {
        "metaphorical_language": "Confused metaphor with literal statement",
        "hypotheticals": "Treated hypothetical scenarios as facts",
    },
}


@dataclass
class ErrorRecord:
    input_text: str
    expected: str
    predicted: str
    category: str
    context: str


class ErrorAnalyzer:
    """Systematic error categorization and analysis."""

    def __init__(self, error_taxonomy: Dict[str, Dict[str, str]] | None = None) -> None:
        self.taxonomy = error_taxonomy or ERROR_TYPES
        self.errors: List[ErrorRecord] = []

    def categorize_error(
        self, input_text: str, expected: str, predicted: str, context: str
    ) -> ErrorRecord:
        category = self._determine_category(input_text, expected, predicted)
        record = ErrorRecord(
            input_text=input_text,
            expected=expected,
            predicted=predicted,
            category=category,
            context=context,
        )
        self.errors.append(record)
        return record

    def _determine_category(self, input_text: str, expected: str, predicted: str) -> str:
        text = input_text.lower()
        if "metaphor" in text or "like" in text:
            return "edge_cases.metaphorical_language"
        if "if" in text or "would" in text:
            return "edge_cases.hypotheticals"
        if expected == "contradiction" and predicted == "consistent":
            if "earlier" in text or "previous" in text:
                return "false_negatives.long_range_dependency"
            return "false_negatives.subtle_contradiction"
        if expected == "consistent" and predicted == "contradiction":
            if "domain" in text:
                return "false_positives.domain_specific"
            return "false_positives.context_needed"
        return "false_positives.creative_ambiguity"

    def summary_by_category(self) -> Dict[str, int]:
        counts = Counter(error.category for error in self.errors)
        return dict(counts)

    def generate_report(self) -> Dict[str, object]:
        counts = self.summary_by_category()
        total = sum(counts.values())
        distribution = {
            category: {"count": count, "percent": (count / total) * 100 if total else 0.0}
            for category, count in counts.items()
        }
        return {
            "total_errors": total,
            "distribution": distribution,
            "examples": [error.__dict__ for error in self.errors[:10]],
        }
