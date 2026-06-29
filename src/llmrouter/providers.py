"""Model providers with a cost model.

Two tiers — a cheap small model and an expensive large model. Offline (no API key) a
deterministic *mock* stands in: the small model answers easy queries correctly and fails
hard ones; the large model answers everything. This makes the cost/quality tradeoff
reproducible without spending a cent. Set ANTHROPIC_API_KEY to route to real Claude tiers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .data import Query

# USD per 1M tokens (Anthropic list prices, blended input+output approximation)
COST_PER_1M = {"small": 0.80, "large": 15.0}      # haiku-class vs opus-class
_REAL_MODEL = {"small": "claude-haiku-4-5-20251001", "large": "claude-opus-4-8"}


@dataclass
class Response:
    correct: bool
    cost_usd: float
    tier: str


def _tokens(prompt: str) -> int:
    # ~1.3 tokens/word in + a modest fixed output budget
    return int(len(prompt.split()) * 1.3) + 120


def _cost(tier: str, prompt: str) -> float:
    return _tokens(prompt) / 1_000_000 * COST_PER_1M[tier]


class MockProvider:
    """Deterministic, key-free. Small model is correct iff the query is easy."""

    def answer(self, tier: str, q: Query) -> Response:
        correct = True if tier == "large" else (q.difficulty == "easy")
        return Response(correct=correct, cost_usd=_cost(tier, q.prompt), tier=tier)


class ClaudeProvider:
    """Real Anthropic call. Grades correctness with the large model as judge."""

    def __init__(self) -> None:
        import anthropic
        self._client = anthropic.Anthropic()

    def answer(self, tier: str, q: Query) -> Response:
        msg = self._client.messages.create(
            model=_REAL_MODEL[tier], max_tokens=512,
            messages=[{"role": "user", "content": q.prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        correct = self._grade(q, text)
        usage = msg.usage
        cost = ((usage.input_tokens + usage.output_tokens) / 1_000_000) * COST_PER_1M[tier]
        return Response(correct=correct, cost_usd=cost, tier=tier)

    def _grade(self, q: Query, answer: str) -> bool:
        verdict = self._client.messages.create(
            model=_REAL_MODEL["large"], max_tokens=5,
            messages=[{"role": "user", "content":
                       f"Question: {q.prompt}\nAnswer: {answer}\n"
                       "Is the answer correct and complete? Reply only YES or NO."}],
        )
        out = "".join(b.text for b in verdict.content if b.type == "text")
        return out.strip().upper().startswith("YES")


def get_provider():
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return ClaudeProvider()
        except Exception:
            pass
    return MockProvider()
