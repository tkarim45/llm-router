# llm-router

Most production LLM traffic is *easy* — greetings, lookups, formatting — yet teams pay
top-model prices for every token. **llm-router** sends each query to the cheapest model that
can still answer it correctly, and **measures** the cost/quality tradeoff against two
baselines instead of asserting it.

```bash
llm-router --mock          # offline, deterministic
llm-router                 # uses real Claude tiers if ANTHROPIC_API_KEY is set
llm-router --json
```

## How it works

1. **Difficulty labels.** A query set tagged `easy` / `hard`, where `hard` means a cheap
   model gets it wrong and you actually need the large one. This is what makes routing
   *measurable*.
2. **Two routers.**
   - `heuristic` — transparent, no training: long prompt or code/math/reasoning cues → large.
   - `learned` — TF-IDF + logistic regression predicts `P(hard)` and routes on a threshold.
3. **Benchmark.** Run `always-small`, `always-large`, and both routers over the same held-out
   queries; report accuracy, total cost (real per-token pricing), and **cost saved vs
   always-large at matched quality**.

## Measured results

`llm-router --mock` on 24 held-out queries (cost model: $0.80/1M small "haiku-class" vs
$15/1M large "opus-class"):

| strategy | accuracy | cost ($) | % → large | vs always-large |
|---|---|---|---|---|
| always-small | 0.500 | 0.00258 | 0% | — |
| always-large | 1.000 | 0.04839 | 100% | — |
| **heuristic router** | **0.958** | **0.02492** | 46% | **−51% cost** |
| learned router | 0.833 | 0.02368 | 46% | −51% cost |

**Recommended: the heuristic router — 96% of always-large quality at 51% of the cost.**

### The honest finding: structure beats vocabulary

The learned TF-IDF router and the heuristic route the *same fraction* to the large model
(46%) and cost the same — but the heuristic is **+12 points more accurate** (95.8% vs 83.3%).
Why: difficulty here is **structural** (length, presence of code/math/multi-step reasoning),
not **lexical**. TF-IDF keys on words, so a held-out hard query phrased with unseen vocabulary
slips through to the small model. Pushing the learned router's threshold down to catch those
recovers quality but erases the savings (at threshold 0.30 it routes everything to large).

The transparent length+keyword heuristic generalizes because it keys on the thing that
actually makes a query hard. The learned router is kept in the benchmark precisely to show
this — a clean reminder that the fancier model isn't automatically the better router.

## Real Claude mode

With `ANTHROPIC_API_KEY` set, the small tier maps to `claude-haiku-4-5` and the large to
`claude-opus-4-8`; answers are graded by the large model as judge, and cost is computed from
real token usage. No key → deterministic mock, so CI and reviewers reproduce the numbers for free.

## Install & test

```bash
pip install -e ".[dev]"      # add ".[dev,claude]" for the real-Claude path
pytest -q                    # 7 passed
```

## Stack

scikit-learn (TF-IDF + logistic regression), pure-Python heuristic router, Anthropic SDK
(optional), real per-token cost model.

## License

MIT
