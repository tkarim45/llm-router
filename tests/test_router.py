from llmrouter.benchmark import compare
from llmrouter.cli import run, split
from llmrouter.data import dataset
from llmrouter.providers import MockProvider
from llmrouter.router import LearnedRouter, heuristic_route


def test_dataset_balanced_and_labeled():
    qs = dataset()
    assert len(qs) >= 40
    assert {q.difficulty for q in qs} == {"easy", "hard"}


def test_mock_small_fails_hard_large_always_right():
    p = MockProvider()
    hard = next(q for q in dataset() if q.difficulty == "hard")
    easy = next(q for q in dataset() if q.difficulty == "easy")
    assert p.answer("small", easy).correct
    assert not p.answer("small", hard).correct
    assert p.answer("large", hard).correct


def test_large_costs_more_than_small():
    p = MockProvider()
    q = dataset()[0]
    assert p.answer("large", q).cost_usd > p.answer("small", q).cost_usd


def test_heuristic_routes_code_to_large():
    code_q = next(q for q in dataset() if "Python function" in q.prompt)
    assert heuristic_route(code_q) == "large"


def test_learned_router_trains_and_separates():
    train, test = split(dataset())
    r = LearnedRouter(threshold=0.35).fit(train)
    # at a recall-tuned threshold it routes most held-out hard queries to large
    hard_test = [q for q in test if q.difficulty == "hard"]
    routed_large = sum(r.route(q) == "large" for q in hard_test)
    assert routed_large / len(hard_test) >= 0.8


def test_heuristic_router_saves_cost_at_high_quality():
    # the structural heuristic is the recommended router: near-large quality, big savings
    res = run(mock=True)
    hr = res["heuristic_router"]
    al = res["always_large"]
    assert hr["cost_savings_vs_large"] > 0.3          # materially cheaper
    assert hr["quality_retained_vs_large"] >= 0.9     # near-large quality
    assert hr["total_cost_usd"] < al["total_cost_usd"]


def test_always_large_perfect_small_lossy():
    res = run(mock=True)
    assert res["always_large"]["accuracy"] == 1.0
    assert res["always_small"]["accuracy"] < 1.0
