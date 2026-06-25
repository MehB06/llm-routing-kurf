# LTT Router — N-model LLM routing

A routing framework that delegates each query to the **cheapest model that is
*certified* safe**, using the Learn-Then-Test (LTT) statistical procedure. Unlike
the other baselines in this benchmark, it has a **formal guarantee**: the
realized regret of the routing rule stays ≤ α with probability ≥ 1−δ.

It generalises the validated two-model proof-of-concept in `baselines/ltt_v1/`
to N models, with a router-agnostic calibration core and a thin adaptor that lets
it stand beside the existing LLMRouterBench baselines on their metrics while
reporting the guarantee column none of them have.

---

## What "certified safe" means

You set a guarantee — *"the routed model must match the best
available model's answer at least (1−α) of the time"* — and LTT calibrates a
decision threshold λ̂ on held-out data so that guarantee provably holds with
probability ≥ 1−δ. Once calibrated, each query goes to the cheapest model whose
score clears λ̂; if none clear it, the query defers to the most capable model.

The framework is **training-free except for the scorer**: the Pareto filter is
combinatorial, the calibration is a hypothesis test (nothing fitted by gradient
descent), and only the injected scorer is a trained ML model.

**The guarantee is conditional on routing.** It bounds the regret *among the
queries actively routed to a cheaper model* — deferrals to the safe fallback are
not under risk by construction.

---

## Architecture

Three pluggable pieces, decoupled so any router can be calibrated by the same
core:

```
RoutingFunction:  f(prompt) -> scores[N]              # trained scorer
LossFn:           L(chosen, correct[N], evaluated[N]) # general N-model regret
CalibrationCore:  {scores, correct, cost, evaluated}  -> certified λ̂  (LTT)
```

Nothing below the `RoutingFunction` boundary knows which router produced the
scores

### The routing rule (cost-ordered, Pareto-aware)

Expensive ≠ better, so the rule does **not** assume the most expensive model is best:

1. **Pareto pre-filter** (offline, on calibration data only): drop any model that
   is dominated (some other model is both cheaper *and* at least as accurate. A
   dominated model can never be the right choice.)
2. **Cost ordering**: sort survivors cheapest → most expensive.
3. **Cheapest-safe routing**: walk survivors cheapest-first, take the first whose
   score clears λ̂; else defer to the most capable *evaluated* model.

LTT certifies the single scalar λ̂ so the realized regret of this whole rule
stays ≤ α with probability ≥ 1−δ.

---

## Package layout

```
baselines/
├── adaptors/
│   └── ltt_adaptor.py        # benchmark bridge: load -> split -> train -> calibrate -> route
└── ltt_router/
    ├── protocols.py          # ModelSpec, QueryRecord, RoutingFunction contracts
    ├── core/                 # the risk-controlled engine (the contribution)
    │   ├── loss.py           # general N-model regret loss
    │   ├── calibration.py    # LTT core: binomial p-value, FST, cheapest-safe rule, λ̂
    │   └── routing.py        # Pareto filter + cost-ordering + the public Router
    ├── routers/              # pluggable scorers (the only trained part)
    │   ├── embedding_lr.py   # the trained scorer (embedding + per-model LR) + caching
    │   └── random_router.py  # ablation control (safety comes from calibration)
    ├── eval/                 # measurement, separate from the engine
    │   ├── metrics.py        # two metric blocks (benchmark-comparable + guarantee)
    │   └── benchmark.py      # emit BaselineRecords + benchmark-aggregator metrics
    ├── splitting.py          # three-way (train/calib/test) prompt-level split
    ├── experiment.py         # repeated-trials harness + α-sweep + figures
    └── tests/                # synthetic data + stub embedder, no downloads
```

---

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Quick start

```bash
# tests — no data or model download needed
python -m pytest baselines/ltt_router/tests/ -v        # 76 passed

# calibrate + route on the benchmark (--verbose prints the calibration loss table)
python -m baselines.adaptors.ltt_adaptor \
    --config config/baseline_config_performance_cost.yaml \
    --alpha 0.15 --seed 42 --verbose

# generate the experiment figures
python -m baselines.ltt_router.experiment \
    --config config/baseline_config_performance_cost.yaml \
    --n-trials 500 \
    --alphas 0.15,0.20,0.25,0.30,0.35,0.40 \
    --outdir experiments
```

The experiment writes three PNGs to `experiments/`:
- `guarantee_histogram.png` — realized-risk distribution over n trials, α line,
  violation rate (should be ≤ δ); +Pareto vs budget-only overlay.
- `alpha_sweep.png` — the cost-savings-vs-risk frontier.
- `benchmark_comparison.png` — our (accuracy, cost) point vs the benchmark
  reference rows and RouteLLM, annotated with the guarantee.

---

## Results

13 flagship models, 10 datasets, 161,520 records; three-way prompt-level split
(60/20/20); δ = 0.10; embedding + per-model logistic-regression scorer. Pareto
survivors at this setting: `deepseek-v3-0324`, `gpt-5`, `qwen3-235b-a22b-2507`.

### The guarantee holds (repeated trials, n = 500)

At α = 0.15, δ = 0.10, realized risk exceeds α on **8.0%** of trials with Pareto
pre-filtering and **6.0%** budget-only — both within the δ = 10% bound. The mass
of the realized-risk distribution sits below the α line. See
`experiments/guarantee_histogram.png`.

### Cost-savings vs risk frontier (α-sweep)

| α    | realized risk | routed % | cost saved |
|------|---------------|----------|------------|
| 0.15 | ~0.115        | ~31%     | ~16%       |
| 0.20 | ~0.19         | ~53%     | ~30%       |
| 0.25 | ~0.225        | ~69%     | ~38%       |
| 0.30 | ~0.265        | ~84%     | ~53%       |
| 0.35 | ~0.34         | ~96%     | ~74%       |
| 0.40 | ~0.39         | ~100%    | ~86%       |

Realized risk stays at or below the α target across the sweep (the guarantee
holding); the routed fraction and cost savings rise as α loosens. See `experiments/alpha_sweep.png`.

### Comparison to baselines

In accuracy/cost space the LTT router sits alongside the strong reference points
(Max Expert, All-Fallback) at accuracy ≈ 0.59, while carrying a **certified risk
bound that no other method reports**. RouteLLM achieves higher accuracy at lower
cost here See
`experiments/benchmark_comparison.png`.