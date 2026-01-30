from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass
class PipelineOutput:
    label: int
    score: float


class NarrativeGuardPipeline:
    """End-to-end pipeline for narrative inconsistency detection."""

    def __init__(self, max_features: int = 30000) -> None:
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.classifier = LogisticRegression(max_iter=200)

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> None:
        features = self.vectorizer.fit_transform(list(texts))
        self.classifier.fit(features, list(labels))

    def predict(self, texts: Iterable[str]) -> List[PipelineOutput]:
        features = self.vectorizer.transform(list(texts))
        scores = self.classifier.predict_proba(features)[:, 1]
        labels = (scores >= 0.5).astype(int)
        return [PipelineOutput(label=int(label), score=float(score)) for label, score in zip(labels, scores)]
