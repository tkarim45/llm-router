"""CLI — train the learned router on a split, benchmark every strategy on held-out queries,
print the cost/quality table."""
from __future__ import annotations

import argparse
import json

from .benchmark import compare
from .data import dataset
from .providers import MockProvider, get_provider
from .router import LearnedRouter


def split(queries, test_frac=0.4, seed=7):
    # deterministic interleaved split keeping easy/hard balance in both halves
    easy = [q for q in queries if q.difficulty == "easy"]
    hard = [q for q in queries if q.difficulty == "hard"]
    n_test_e = int(len(easy) * test_frac)
    n_test_h = int(len(hard) * test_frac)
    test = easy[:n_test_e] + hard[:n_test_h]
    train = easy[n_test_e:] + hard[n_test_h:]
    return train, test


def run(threshold: float = 0.5, mock: bool = False) -> dict:
    qs = dataset()
    train, test = split(qs)
    router = LearnedRouter(threshold=threshold).fit(train)
    provider = MockProvider() if mock else get_provider()
    results = compare(test, router, provider)
    results["_meta"] = {"train_n": len(train), "test_n": len(test),
                        "provider": type(provider).__name__, "threshold": threshold}
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark LLM routing vs always-small/always-large.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--mock", action="store_true", help="force offline mock provider")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    res = run(threshold=args.threshold, mock=args.mock)
    if args.json:
        print(json.dumps(res, indent=2))
        return

    print("=" * 70)
    print(f"  LLM ROUTER BENCHMARK   ({res['_meta']['provider']}, "
          f"{res['_meta']['test_n']} held-out queries)")
    print("=" * 70)
    hdr = f"{'strategy':<18}{'accuracy':>10}{'cost($)':>12}{'%->large':>10}{'vs-large':>12}"
    print(hdr)
    print("-" * 70)
    for name in ("always_small", "always_large", "heuristic_router", "learned_router"):
        r = res[name]
        save = f"-{r['cost_savings_vs_large']:.0%}" if "cost_savings_vs_large" in r else "  —"
        print(f"{name:<18}{r['accuracy']:>10.3f}{r['total_cost_usd']:>12.5f}"
              f"{r['large_call_fraction']:>10.0%}{save:>12}")
    print("-" * 70)
    best = res["recommended"]
    br = res[best]
    print(f"recommended: {best} — {br['quality_retained_vs_large']:.0%} of always-large "
          f"quality at {1 - br['cost_savings_vs_large']:.0%} of the cost")
    print("=" * 70)


if __name__ == "__main__":
    main()
