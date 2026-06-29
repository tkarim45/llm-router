"""Model providers with a cost model.

Two tiers — a cheap small model and an expensive large model. Three backends:

  * MockProvider    — deterministic, key-free. Small model answers easy queries correctly and
                      fails hard ones; large answers everything. Reproduces the README numbers
                      for free.
  * BedrockProvider — real Claude on AWS Bedrock (the repo convention). Reads AWS creds from
                      the environment / .env; small=Haiku, large=Opus by default.
  * ClaudeProvider  — real Claude via the first-party Anthropic API (ANTHROPIC_API_KEY).

Selection is automatic (see get_provider) or forced with LLM_ROUTER_PROVIDER /
the CLI --provider flag.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .data import Query

# Load .env if present (AWS creds / model overrides) — best-effort, never fatal.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv optional
    pass

# USD per 1M tokens (blended input+output approximation) used for the cost comparison.
COST_PER_1M = {"small": 0.80, "large": 15.0}  # haiku-class vs opus-class

# First-party Anthropic API model IDs.
_API_MODEL = {"small": "claude-haiku-4-5-20251001", "large": "claude-opus-4-8"}

# AWS Bedrock inference-profile IDs (repo convention: "global." for max availability).
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
_BEDROCK_MODEL = {
    "small": os.getenv("BEDROCK_SMALL_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0"),
    "large": os.getenv("BEDROCK_LARGE_MODEL", "global.anthropic.claude-opus-4-6-v1"),
}


@dataclass
class Response:
    correct: bool
    cost_usd: float
    tier: str


def _tokens(prompt: str) -> int:
    # ~1.3 tokens/word in + a modest fixed output budget (mock cost only)
    return int(len(prompt.split()) * 1.3) + 120


def _mock_cost(tier: str, prompt: str) -> float:
    return _tokens(prompt) / 1_000_000 * COST_PER_1M[tier]


class MockProvider:
    """Deterministic, key-free. Small model is correct iff the query is easy."""

    def answer(self, tier: str, q: Query) -> Response:
        correct = True if tier == "large" else (q.difficulty == "easy")
        return Response(correct=correct, cost_usd=_mock_cost(tier, q.prompt), tier=tier)


class _AnthropicLikeProvider:
    """Shared logic for the real backends: ask a tier, grade with the large model as judge,
    cost from real token usage. Subclasses set self._client and self._models."""

    _client = None
    _models: dict = {}
    _create_kwargs: dict = {}

    def answer(self, tier: str, q: Query) -> Response:
        msg = self._client.messages.create(
            model=self._models[tier], max_tokens=512,
            messages=[{"role": "user", "content": q.prompt}],
            **self._create_kwargs,
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        correct = self._grade(q, text)
        usage = msg.usage
        cost = ((usage.input_tokens + usage.output_tokens) / 1_000_000) * COST_PER_1M[tier]
        return Response(correct=correct, cost_usd=cost, tier=tier)

    def _grade(self, q: Query, answer: str) -> bool:
        verdict = self._client.messages.create(
            model=self._models["large"], max_tokens=5,
            messages=[{"role": "user", "content":
                       f"Question: {q.prompt}\nAnswer: {answer}\n"
                       "Is the answer correct and complete? Reply only YES or NO."}],
            **self._create_kwargs,
        )
        out = "".join(b.text for b in verdict.content if b.type == "text")
        return out.strip().upper().startswith("YES")


class BedrockProvider(_AnthropicLikeProvider):
    """Real Claude on AWS Bedrock (repo convention). Creds from AWS_* env vars / .env."""

    def __init__(self) -> None:
        from anthropic import AnthropicBedrock

        self._client = AnthropicBedrock(aws_region=AWS_REGION)
        self._models = _BEDROCK_MODEL
        self._create_kwargs = {"temperature": 0.0}  # reproducible; accepted on Haiku/Opus 4.6


class ClaudeProvider(_AnthropicLikeProvider):
    """Real Claude via the first-party Anthropic API (ANTHROPIC_API_KEY)."""

    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self._models = _API_MODEL
        self._create_kwargs = {}  # Opus 4.8 rejects temperature


def _has_aws_creds() -> bool:
    return bool(
        os.getenv("AWS_ACCESS_KEY_ID")
        or os.getenv("AWS_PROFILE")
        or os.getenv("AWS_SESSION_TOKEN")
        or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    )


def get_provider(name: str = "auto"):
    """Pick a backend. name: auto | mock | bedrock | anthropic.

    auto → Bedrock if AWS creds present, else Anthropic API if ANTHROPIC_API_KEY,
    else the deterministic mock. Real backends fall back to mock on construction error.
    """
    name = (os.getenv("LLM_ROUTER_PROVIDER") or name or "auto").lower()
    if name == "mock":
        return MockProvider()
    if name == "bedrock":
        try:
            return BedrockProvider()
        except Exception:
            return MockProvider()
    if name == "anthropic":
        try:
            return ClaudeProvider()
        except Exception:
            return MockProvider()
    # auto
    if _has_aws_creds():
        try:
            return BedrockProvider()
        except Exception:
            pass
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return ClaudeProvider()
        except Exception:
            pass
    return MockProvider()
