# LTT Router — N-model LLM routing

A routing framework that delegates each query to the **cheapest model that is
*certified* safe**, using the Learn-Then-Test (LTT) statistical procedure. Unlike
the other baselines in this benchmark, it has a **formal guarantee**: LTT gives a
high-probability bound on the *population* regret of the calibrated routing rule
— with probability ≥ 1−δ over the calibration draw, the rule's true regret is
≤ α. (A finite test split is only a diagnostic and may sit above α by sampling
noise without violating the guarantee.)

It generalises a validated two-model proof-of-concept to N models, with a
router-agnostic calibration core and a thin adaptor that lets it stand beside the
existing LLMRouterBench baselines on their metrics while reporting the guarantee
column none of them have.

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
not under risk by construction. The conditioning event depends on λ, so the risk
need not be monotone in λ; FST does not require monotonicity. The `min_routed`
power floor filters λ's on their routed *count*, which is a function of the
calibration scores only (not the outcomes), so conditioning on the scores fixes
the whole FST sequence and family-wise error control is preserved — see the
docstring in `core/calibration.py` for the full argument, and
`tests/test_ltt_core.py::test_fwer_guarantee_holds_empirically` for a
Monte-Carlo check of the promise on a model with closed-form risk.

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

1. **Pareto pre-filter** (offline, on the **design split** — carved from train,
   disjoint from calibration): drop any model that is dominated (some other model
   is both cheaper *and* at least as accurate. A dominated model can never be the
   right choice.) Designing the action set on train keeps the calibration labels
   reserved solely for the LTT hypothesis test. The filter is a heuristic
   pre-selection on point estimates; the guarantee certifies the *resulting*
   rule, whatever the selection was.
2. **Cost ordering**: sort survivors cheapest → most expensive.
3. **Cheapest-safe routing**: walk survivors cheapest-first, take the first whose
   score clears λ̂; else defer to the most capable *evaluated* model.

LTT certifies the single scalar λ̂ so the *population* regret of this whole rule
is ≤ α with probability ≥ 1−δ.

---

## Package layout

```
baselines/
├── adaptors/
│   └── ltt_adaptor.py        # benchmark bridge: load -> split -> train -> design+calibrate -> route
└── ltt_router/
    ├── protocols.py          # ModelSpec, QueryRecord, RoutingFunction contracts
    ├── core/                 # the risk-controlled engine (the contribution)
    │   ├── loss.py           # general N-model regret loss
    │   ├── calibration.py    # LTT core: binomial p-value, FST, cheapest-safe rule, λ̂
    │   └── routing.py        # Pareto filter + cost-ordering + the public Router
    ├── routers/              # pluggable scorers (the only trained part)
    │   ├── embedding_lr.py   # the trained scorer (embedding + per-model LR) + caching
    │   └── routellm_mf.py    
    ├── routellm/             # the optional RouteLLM integration (scripts, config, docs)
    │   ├── pipeline.py       # offline prep: build-inputs + fit-calibrators
    │   ├── run_experiment.py # single-seed α-sweep (table + JSON) + benchmark point
    │   └── train_config.json # MF training config (dim=384, use_proj=false)
    ├── eval/                 # measurement, separate from the engine
    │   ├── metrics.py        # two metric blocks (benchmark-comparable + guarantee)
    │   └── benchmark.py      # emit BaselineRecords + benchmark-aggregator metrics
    ├── splitting.py          # three-way (train/calib/test) prompt-level split
    ├── experiment.py         # repeated-trials harness + α-sweep + figures + JSONs
    └── tests/                # synthetic data + stub embedder, no downloads

results/ltt_router/           # generated figures + the JSONs of their numbers
```

Torch is optional: the core LTT path (`core/`, `routers/embedding_lr.py`, the
adaptor) and even `routers/routellm_mf.py`'s calibration utilities import
without it (the torch-dependent pieces — checkpoint loading and RouteLLM's
model-ID table — are behind lazy imports). Torch is required only to *train or
load* the MF checkpoint, and `tests/test_routellm_mf.py` skips itself cleanly
when torch is absent.

---

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Quick start

```bash
# tests — no benchmark data or model download needed. The minimal environment is
#   pip install numpy scipy scikit-learn matplotlib loguru pytest
# (the full requirements.txt also works; without torch the MF scorer tests are
# skipped, everything else runs).
python -m pytest baselines/ltt_router/tests/ -v

# calibrate + route on the benchmark (--verbose prints the calibration loss table)
python -m baselines.adaptors.ltt_adaptor \
    --config config/baseline_config_performance_cost.yaml \
    --alpha 0.15 --seed 42 --verbose

# generate the experiment figures + JSONs (embedding-LR scorer)
python -m baselines.ltt_router.experiment \
    --config config/baseline_config_performance_cost.yaml \
    --n-trials 500 \
    --alphas 0.15,0.20,0.25,0.30,0.35,0.40 \
    --outdir results/ltt_router
```

The experiment writes three PNGs to `results/ltt_router/`, each with a JSON of
the numbers behind it (`guarantee_histogram.json`, `alpha_sweep.json`,
`benchmark_comparison.json` — timestamp, git commit, CLI args, metrics). Numbers
quoted in this README and in the paper are copied from those JSONs.

- `guarantee_histogram.png` — realized-risk distribution over the *certifying*
  trials, α line, pooled test-risk estimate and corrected-violation rate
  (should be ≤ δ); abstaining trials are counted separately, never plotted as
  zero-risk mass.
- `alpha_sweep.png` — the cost-savings-vs-risk frontier, with routed and
  genuinely-delegated fractions per α; abstaining α's are shaded.
- `benchmark_comparison.png` — our (accuracy, cost) point vs the benchmark
  reference rows and RouteLLM, annotated with the guarantee.

---

## Results

13 flagship models, 10 datasets, 161,520 records; prompt-level split into
train / calibration / test (60/20/20). The action set (Pareto survivors,
fallback) is designed on a split carved from train, so calibration labels are
used only for the LTT test. δ = 0.10; embedding + per-model logistic-regression
scorer. Pareto survivors at this setting: `deepseek-v3-0324`, `gpt-5`,
`qwen3-235b-a22b-2507`.

### The guarantee holds (repeated trials, n = 500)

At α = 0.15, δ = 0.10: pooled test-risk estimate **0.133 ≤ α** (margin +0.017)
and corrected violation rate **1.7% ≤ δ = 10%**. 484 of 500 trials certified;
the 16 abstaining trials are reported separately and excluded from the
histogram (an abstention routes nothing and is never a violation). The mass of
the realized-risk distribution sits below the α line. See
`results/ltt_router/guarantee_histogram.png` (+ `.json`).

### Cost-savings vs risk frontier (α-sweep)

Realized risk stays at or below the α target across the sweep (the guarantee
holding); the routed fraction and cost savings rise as α loosens. See
`results/ltt_router/alpha_sweep.png` (+ `.json`).

### Comparison to baselines

In accuracy/cost space the LTT router sits alongside the strong reference points
(Max Expert, All-Fallback) at accuracy ≈ 0.59, while carrying a **certified risk
bound that no other method reports**. RouteLLM achieves higher unconstrained
accuracy at lower cost here; see the RouteLLM section below for the head-to-head
under the guarantee, and `results/ltt_router/benchmark_comparison.png`.

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
routers/routellm_mf.py        # MF scorer + built-in calibration (Platt/isotonic)
routellm/pipeline.py          # offline prep: build-inputs + fit-calibrators
routellm/run_experiment.py    # single-seed α-sweep (table + JSON) + benchmark point
routellm/train_config.json    # MF training config
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
scorer is always calibrated: per-model **Platt** and **isotonic** calibrators are
fit on the held-out `train_cal` slice, and ONE method is chosen for the whole
run by **mean cross-validated ECE** over the non-degenerate models (in-sample
ECE would let isotonic overfit to a fake ~0). Mean ECE drops from ~0.075 (raw
`sigmoid(δ)`) to ~0.03 (calibrated).

Note the reliability diagram (`reliability.png`) is a **secondary** diagnostic of
the scorer's probability calibration. It is *not* the risk guarantee — the
guarantee comes from LTT calibration on held-out data, not from probability
calibration. Probability calibration only improves routing efficiency (how many
queries can safely clear a useful λ); it does not itself control risk.

### The certification frontier

The cheapest Pareto survivor (`deepseek-v3-0324`) is only ≈40% accurate, so at
**strict α the calibrated MF scorer correctly abstains** — no threshold is both
safe and adequately powered, and the router defers everything (realized
risk = 0 because nothing is routed, *not* because routing is perfect). It begins
to certify around **α ≈ 0.25**, where it routes ~95% with a certified bound. The
sweep is reported as a table + `alpha_sweep_routellm.json` (with so few
certifying α's it makes a weak figure, so no PNG is generated for it).

At the α = 0.25 benchmark point (`benchmark_comparison_routellm.png` + `.json`):
accuracy **≈ 0.607** at **≈ $0.024**/query, vs RouteLLM's published **≈ 0.615**
at **≈ $0.033** — slightly below their unconstrained accuracy at ~27% lower
cost, while carrying the certified bound (`population regret ≤ 0.25 with
probability ≥ 0.9`) that none of the baselines report. The small accuracy gap is
the price of the guarantee and of stretching a pairwise-trained `δ` across all
N models.

### Reproduce

```bash
# 1. build MF inputs (TRAIN -> train_fit/train_cal, pairwise + embeddings)
python -m baselines.ltt_router.routellm.pipeline build-inputs \
    --config config/baseline_config_performance_cost.yaml \
    --out-dir baselines/RouteLLM/data/ltt_perfcost_seed42 --seed 42

# 2. train MF (RouteLLM's trainer, unedited; dim=384, use_proj=false)
python -m baselines.RouteLLM.routers.matrix_factorization.train_matrix_factorization \
    --config baselines/ltt_router/routellm/train_config.json

# 3. fit per-model calibrators (CV-ECE picks Platt/isotonic)
python -m baselines.ltt_router.routellm.pipeline fit-calibrators \
    --config config/baseline_config_performance_cost.yaml \
    --artifacts-dir baselines/RouteLLM/data/ltt_perfcost_seed42 \
    --checkpoint baselines/RouteLLM/checkpoints/ltt_perfcost_seed42/mf_model.pt \
    --out baselines/RouteLLM/checkpoints/ltt_perfcost_seed42/calibrators.pkl \
    --seed 42 --diagram baselines/RouteLLM/checkpoints/ltt_perfcost_seed42/reliability.png

# 4. α-sweep (table + JSON) + benchmark point (auto-picks first certifying α)
python -m baselines.ltt_router.routellm.run_experiment \
    --config config/baseline_config_performance_cost.yaml \
    --mf-checkpoint baselines/RouteLLM/checkpoints/ltt_perfcost_seed42/mf_model.pt \
    --mf-calibrators baselines/RouteLLM/checkpoints/ltt_perfcost_seed42/calibrators.pkl \
    --alpha 0.25 --alphas 0.15,0.20,0.22,0.25,0.30,0.35 \
    --outdir results/ltt_router
```

Writes `alpha_sweep_routellm.json` (certification frontier) and
`benchmark_comparison_routellm.png` + `.json` (at the first certifying α).