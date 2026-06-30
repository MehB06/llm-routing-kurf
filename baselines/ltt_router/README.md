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

---

## RouteLLM matrix-factorization scorer under the LTT guarantee

Beyond the embedding+LR scorer, the framework can wrap **RouteLLM's trained
matrix-factorization (MF) scorer** as the `RoutingFunction`, certifying *its*
scores with the same LTT core. This isolates the LTT wrapper's cost: same scorer
RouteLLM uses, now carrying a guarantee.

### What is reused vs added

We reuse RouteLLM's **trained per-model quality score** `δ(M, q)`, not its
2-model routing wrapper. RouteLLM's pairwise winrate is `σ(δ(a) − δ(b))`; `δ`
itself is per-model, so we read one `δ` per model and calibrate it to a success
probability. Everything below the `RoutingFunction` boundary (split, Pareto, FST,
metrics) is untouched.

```
baselines/ltt_router/routers/routellm_mf.py   # MF scorer + built-in calibration (Platt/isotonic)
baselines/ltt_router/routers/mf_pipeline.py   # offline prep: build-inputs + fit-calibrators
baselines/ltt_router/plot_routellm_sweep.py   # honest α-sweep (certified vs abstained)
baselines/ltt_router/run_routellm_experiment.py  # single-seed α-sweep + benchmark point
```

### Why single-seed (seed 42)

The MF checkpoint is trained **once** on seed-42's `train_fit` and the
calibrators on seed-42's `train_cal`. The multi-seed guarantee histogram
re-splits per seed, which would leak seed-42 training prompts into other seeds'
CALIB/TEST and break disjointness. At seed 42 every set is disjoint by
construction: `train_fit → MF`, `train_cal → calibrators`, `CALIB → LTT`,
`TEST → eval`. So the MF scorer is reported via an α-sweep + benchmark point at
seed 42; the LR scorer keeps its full multi-seed histogram.

### Calibration is built in 

Raw `δ` has the right *ordering* but is not a success probability, so the MF
scorer is always calibrated: per-model **Platt** and **isotonic** are fit on the
held-out `train_cal` slice and the better is chosen by **cross-validated ECE**
(in-sample ECE would let isotonic overfit to a fake ~0). Mean ECE drops from
~0.075 (raw `sigmoid(δ)`) to ~0.03 (calibrated).

### The certification frontier

The cheapest Pareto survivor (`deepseek-v3-0324`) is only ≈40% accurate, so at
**strict α the calibrated MF scorer correctly abstains** — no threshold is both
safe and adequately powered, and the router defers everything (realized
risk = 0 because nothing is routed, *not* because routing is perfect). It begins
to certify around **α ≈ 0.25**, where it routes ~95% with a certified bound.

This is a principled, reportable result, not a failure: it characterizes the MF
scorer in the LTT framework's own terms. The LTT-Router trades raw accuracy for
a *certified* bound — it sits slightly below RouteLLM's unconstrained accuracy
(~0.585 vs ~0.615) because the guarantee forces conservatism, while carrying the
risk-control column none of the baselines report.

### Reproduce

```bash
# 1. build MF inputs (TRAIN -> train_fit/train_cal, pairwise + embeddings)
python -m baselines.ltt_router.routers.mf_pipeline build-inputs \
    --config config/baseline_config_performance_cost.yaml \
    --out-dir baselines/RouteLLM/data/ltt_perfcost_seed42 --seed 42

# 2. train MF (RouteLLM's trainer, unedited; dim=384, use_proj=false)
python -m baselines.RouteLLM.routers.matrix_factorization.train_matrix_factorization \
    --config baselines/RouteLLM/mf_train_config_ltt.json

# 3. fit per-model calibrators (CV-ECE picks Platt/isotonic)
python -m baselines.ltt_router.routers.mf_pipeline fit-calibrators \
    --config config/baseline_config_performance_cost.yaml \
    --artifacts-dir baselines/RouteLLM/data/ltt_perfcost_seed42 \
    --checkpoint baselines/RouteLLM/checkpoints/ltt_perfcost_seed42/mf_model.pt \
    --out baselines/RouteLLM/checkpoints/ltt_perfcost_seed42/calibrators.pkl \
    --seed 42 --diagram baselines/RouteLLM/checkpoints/ltt_perfcost_seed42/reliability.png

# 4. α-sweep + benchmark point (auto-picks first certifying α)
python -m baselines.ltt_router.run_routellm_experiment \
    --config config/baseline_config_performance_cost.yaml \
    --mf-checkpoint baselines/RouteLLM/checkpoints/ltt_perfcost_seed42/mf_model.pt \
    --mf-calibrators baselines/RouteLLM/checkpoints/ltt_perfcost_seed42/calibrators.pkl \
    --alpha 0.25 --alphas 0.15,0.20,0.22,0.25,0.30,0.35 \
    --outdir baselines/ltt_router/experiments
```

Writes `alpha_sweep_routellm.png` (certification frontier) and
`benchmark_comparison_routellm.png` (at the first certifying α).