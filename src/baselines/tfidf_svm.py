from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


@dataclass
class TFIDFPrediction:
    label: int
    score: float


class TFIDFBaseline:
    """Classical TF-IDF + linear SVM baseline."""

    def __init__(self, max_features: int = 20000, min_df: int = 2) -> None:
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=max_features, min_df=min_df)),
                ("clf", LinearSVC())
            ]
        )

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> None:
        self.pipeline.fit(list(texts), list(labels))

    def predict(self, texts: Iterable[str]) -> List[int]:
        return self.pipeline.predict(list(texts)).tolist()

    def decision_function(self, texts: Iterable[str]) -> List[float]:
        return self.pipeline.decision_function(list(texts)).tolist()

    def batch_predict(self, texts: Iterable[str]) -> List[TFIDFPrediction]:
        labels = self.predict(texts)
        scores = self.decision_function(texts)
        return [TFIDFPrediction(label=label, score=score) for label, score in zip(labels, scores)]
