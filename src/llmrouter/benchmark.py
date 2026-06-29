"""Benchmark routing strategies against two baselines on the same held-out queries:
always-small (cheapest, lossy), always-large (perfect, expensive), and the routers.

The headline numbers: accuracy (did we get it right?), total cost, and cost saved vs
always-large at equal-or-near quality.
"""
from __future__ import annotations

from .data import Query
from .providers import Response


def _run(strategy, queries: list[Query], provider) -> dict:
    """strategy(q) -> tier. Returns accuracy + total/avg cost over the query set."""
    responses: list[Response] = []
    for q in queries:
        tier = strategy(q)
        responses.append(provider.answer(tier, q))
    n = len(responses)
    correct = sum(r.correct for r in responses)
    total_cost = sum(r.cost_usd for r in responses)
    large_calls = sum(r.tier == "large" for r in responses)
    return {
        "accuracy": round(correct / n, 4),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_query_usd": round(total_cost / n, 8),
        "large_call_fraction": round(large_calls / n, 4),
        "n": n,
    }


def compare(test: list[Query], learned_router, provider) -> dict:
    from .router import heuristic_route

    always_small = _run(lambda q: "small", test, provider)
    always_large = _run(lambda q: "large", test, provider)
    heuristic = _run(heuristic_route, test, provider)
    learned = _run(learned_router.route, test, provider)

    big = always_large["total_cost_usd"]
    for r in (heuristic, learned):
        r["cost_savings_vs_large"] = round(1 - r["total_cost_usd"] / big, 4) if big else 0.0
        r["quality_retained_vs_large"] = round(
            r["accuracy"] / always_large["accuracy"], 4) if always_large["accuracy"] else 0.0

    # recommend the router that keeps quality high, then maximizes savings
    candidates = {"heuristic_router": heuristic, "learned_router": learned}
    safe = {k: v for k, v in candidates.items() if v["quality_retained_vs_large"] >= 0.95}
    pool = safe or candidates
    best = max(pool, key=lambda k: pool[k]["cost_savings_vs_large"])

    return {
        "always_small": always_small,
        "always_large": always_large,
        "heuristic_router": heuristic,
        "learned_router": learned,
        "recommended": best,
    }
