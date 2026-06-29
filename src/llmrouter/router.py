"""The router: decide small vs large *before* spending money on the large model.

A learned router (TF-IDF + logistic regression) predicts whether a query is hard; if the
predicted probability of "hard" exceeds a threshold it routes to the large model, else the
small one. A transparent heuristic router (length + code/math/reasoning cues) is included as
a no-training baseline.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import Query

_CODE_MATH = re.compile(
    r"\b(function|code|algorithm|sql|prove|derive|complexity|matrix|probability|"
    r"recursive|implement|design|cache|theorem|paradox|equation|solve)\b", re.I)


def heuristic_route(q: Query, word_threshold: int = 18) -> str:
    """No-training router: long prompts or code/math/reasoning cues -> large."""
    words = len(q.prompt.split())
    if words >= word_threshold or _CODE_MATH.search(q.prompt):
        return "large"
    return "small"


class LearnedRouter:
    """TF-IDF + logistic regression trained on labeled difficulties."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])

    def fit(self, queries: list[Query]) -> "LearnedRouter":
        X = [q.prompt for q in queries]
        y = [1 if q.difficulty == "hard" else 0 for q in queries]
        self.model.fit(X, y)
        return self

    def prob_hard(self, q: Query) -> float:
        return float(self.model.predict_proba([q.prompt])[0][1])

    def route(self, q: Query) -> str:
        return "large" if self.prob_hard(q) >= self.threshold else "small"
