"""
Experiment 

Three experiments, each writing a PNG into an output dir 

  1. REPEATED-TRIALS GUARANTEE 
     Run the whole pipeline n_trials times from scratch (fresh seed -> fresh
     split -> recalibrate -> re-evaluate) and plot the distribution of realized
     TEST risk.

     IMPORTANT: the LTT guarantee bounds the TRUE (population) risk, NOT the
     noisy finite-sample test risk. A fresh test draw scatters around the true
     risk, so comparing each trial's test risk to alpha is NOT a valid check (it
     sits near 50% even when every certified threshold is genuinely safe). We
     therefore validate with:
       - pooled_true_risk: pool routed queries across all trials -> low-variance
         estimate of the true risk the guarantee actually bounds (must be ≤ α).
       - corrected_violation_rate: count a trial only when its test data shows,
         with confidence, that its true risk exceeds α (must stay ≤ δ).
     raw_violation_rate is retained as a DIAGNOSTIC only.

  2. ALPHA-SWEEP (cost-savings vs risk frontier).
     Sweep alpha; for each, plot realized risk and routed/cost-saved. 

  3. BENCHMARK COMPARISON.
     Plot our (accuracy, cost) point against the existing baselines' published
     numbers (evaluate.load_baseline_results), annotated with our guarantee 
     the column none of them report.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from baselines.ltt_router.eval.metrics import evaluate, EvalResult, load_baseline_results


DEFAULT_OUTDIR = "baselines/ltt_router/experiments"


def _ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


# 1. Repeated-trials guarantee histogram
@dataclass
class TrialOutcome:
    realized_risk: float          # per-trial conditional risk (noisy, finite test draw)
    routed_fraction: float
    certified: bool
    cost_saved: float
    n_routed: int = 0             # routed test queries this trial (risk denominator)
    n_routed_fail: int = 0        # regret events among them (risk numerator)


def run_repeated_trials(
    make_result: Callable[[int, bool], object],
    n_trials: int = 200,
    apply_pareto: bool = True,
    progress: bool = True,
    label: str = "",
) -> List[TrialOutcome]:
    """
    make_result(seed, apply_pareto) -> AdaptorResult. We call it n_trials times
    with different seeds, evaluate each, and collect the realized-risk outcomes.
    Kept abstract so the caller wires in either synthetic or real data.

    progress: print a per-seed line with elapsed time + ETA, so a long run shows
    how far along it is instead of hanging silently.
    """
    import time

    outcomes = []
    t0 = time.time()
    for t in range(n_trials):
        result = make_result(t, apply_pareto)
        ev = evaluate(result)
        outcomes.append(TrialOutcome(
            realized_risk=ev.realized_risk,
            routed_fraction=ev.routed_fraction,
            certified=ev.certified,
            cost_saved=ev.cost_saved,
            n_routed=ev.n_routed,
            n_routed_fail=ev.n_routed_fail,
        ))
        if progress:
            done = t + 1
            elapsed = time.time() - t0
            rate = elapsed / done
            eta = rate * (n_trials - done)
            print(
                f"    [{label}] seed {done:>3}/{n_trials}  "
                f"risk={ev.realized_risk:.3f} routed={ev.routed_fraction:.0%} "
                f"cert={ev.certified}  | {elapsed:5.1f}s elapsed, ETA {eta:5.1f}s",
                flush=True,
            )
    return outcomes


def raw_violation_rate(outcomes: List[TrialOutcome], alpha: float) -> float:
    """
    DIAGNOSTIC ONLY — fraction of trials whose *finite-sample* test risk exceeds
    alpha. This is NOT the guarantee: the LTT promise bounds the TRUE (population)
    risk, and a fresh finite test draw scatters around it, so this can sit near
    50% even when every certified threshold is genuinely safe. Use
    pooled_true_risk / corrected_violation_rate to judge the guarantee.
    """
    risks = np.array([o.realized_risk for o in outcomes])
    return float(np.mean(risks > alpha)) if len(risks) else 0.0


def pooled_true_risk(outcomes: List[TrialOutcome]) -> float:
    """
    Estimate the TRUE risk of the certified rule by pooling routed queries across
    ALL trials (total regret events / total routed queries). With n=500 trials
    this denominator is large, so this is a low-variance estimate of the
    population risk the guarantee actually bounds. The guarantee holds iff this
    pooled estimate is ≤ alpha.
    """
    nf = sum(o.n_routed_fail for o in outcomes)
    nn = sum(o.n_routed for o in outcomes)
    return float(nf / nn) if nn else 0.0


def _risk_lower_bound(n_fail: int, n: int, delta: float) -> float:
    """(1 − delta) Clopper–Pearson LOWER bound on risk from (n_fail, n)."""
    from scipy import stats
    if n == 0:
        return 0.0
    if n_fail == 0:
        return 0.0
    return float(stats.beta.ppf(delta, n_fail, n - n_fail + 1))


def corrected_violation_rate(
    outcomes: List[TrialOutcome], alpha: float, delta: float
) -> float:
    """
    Honest, denominator-aware violation rate. A finite test draw scatters around
    the true risk, so a trial sitting *noisily* above alpha is not a violation.
    """
    if not outcomes:
        return 0.0
    flags = []
    for o in outcomes:
        if o.n_routed == 0:
            continue
        flags.append(_risk_lower_bound(o.n_routed_fail, o.n_routed, delta) > alpha)
    return float(np.mean(flags)) if flags else 0.0


# Backwards-compatible alias (old name) — points at the diagnostic, not the test.
violation_rate = raw_violation_rate


def plot_guarantee_histogram(
    outcomes_pareto: List[TrialOutcome],
    outcomes_budget: Optional[List[TrialOutcome]],
    alpha: float,
    delta: float,
    outdir: str = DEFAULT_OUTDIR,
    fname: str = "guarantee_histogram.png",
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_outdir(outdir)
    fig, ax = plt.subplots(figsize=(8, 5))

    rp = np.array([o.realized_risk for o in outcomes_pareto])
    true_p = pooled_true_risk(outcomes_pareto)
    cv_p = corrected_violation_rate(outcomes_pareto, alpha, delta)
    ax.hist(rp, bins=30, alpha=0.6,
            label=f"+Pareto (true risk {true_p:.3f}, viol {cv_p:.1%})", color="#2a7ae2")

    if outcomes_budget is not None:
        rb = np.array([o.realized_risk for o in outcomes_budget])
        true_b = pooled_true_risk(outcomes_budget)
        cv_b = corrected_violation_rate(outcomes_budget, alpha, delta)
        ax.hist(rb, bins=30, alpha=0.5,
                label=f"budget-only (true risk {true_b:.3f}, viol {cv_b:.1%})", color="#e2802a")

    ax.axvline(alpha, color="red", linestyle="--", linewidth=2, label=f"α = {alpha}")
    ax.axvline(true_p, color="#2a7ae2", linestyle=":", linewidth=2,
               label=f"pooled true risk = {true_p:.3f}")
    ax.set_xlabel("realized risk on test split (per trial)")
    ax.set_ylabel("number of trials")
    ax.set_title(f"Repeated-trials guarantee (n={len(outcomes_pareto)}, δ={delta})\n"
                 f"true risk ≤ α and corrected violations ≤ δ = {delta:.0%}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# 2. Alpha-sweep frontier
def run_alpha_sweep(
    make_result_at_alpha: Callable[[float], object],
    alphas: Sequence[float],
) -> List[EvalResult]:
    return [evaluate(make_result_at_alpha(a)) for a in alphas]


def plot_alpha_sweep(
    evals: List[EvalResult],
    alphas: Sequence[float],
    outdir: str = DEFAULT_OUTDIR,
    fname: str = "alpha_sweep.png",
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_outdir(outdir)
    realized = [e.realized_risk for e in evals]
    routed = [e.routed_fraction for e in evals]
    saved = [e.cost_saved for e in evals]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(alphas, realized, "o-", color="#d62728", label="realized risk")
    ax1.plot(alphas, alphas, "--", color="gray", alpha=0.6, label="α (target)")
    ax1.set_xlabel("α (risk target)")
    ax1.set_ylabel("realized risk", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")

    ax2 = ax1.twinx()
    ax2.plot(alphas, routed, "s-", color="#2a7ae2", label="routed fraction")
    ax2.plot(alphas, saved, "^-", color="#2ca02c", label="cost saved")
    ax2.set_ylabel("routed fraction / cost saved", color="#2a7ae2")
    ax2.tick_params(axis="y", labelcolor="#2a7ae2")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("α-sweep: cost-savings vs risk frontier")
    fig.tight_layout()
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# 3. Benchmark comparison
def plot_benchmark_comparison(
    ours: EvalResult,
    repo_root: str = ".",
    outdir: str = DEFAULT_OUTDIR,
    fname: str = "benchmark_comparison.png",
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_outdir(outdir)
    baselines = load_baseline_results(repo_root)

    fig, ax = plt.subplots(figsize=(8, 5))
    # Reference points from our test set (accuracy vs cost).
    for name, m in ours.reference.items():
        if not np.isnan(m.get("cost", np.nan)):
            ax.scatter(m["cost"], m["accuracy"], marker="x", color="gray")
            ax.annotate(name, (m["cost"], m["accuracy"]), fontsize=8, color="gray")

    # Existing baselines (published numbers).
    for name, m in baselines.items():
        ax.scatter(m["cost"], m["accuracy"], marker="o", s=60)
        ax.annotate(name, (m["cost"], m["accuracy"]), fontsize=9)

    # Our router -- highlighted, annotated with the guarantee.
    ax.scatter(ours.avg_cost, ours.avg_accuracy, marker="*", s=250, color="#d62728",
               zorder=5, label="LTT-Router")
    ax.annotate(f"LTT-Router\n(risk≤{ours.alpha}, certified={ours.certified})",
                (ours.avg_cost, ours.avg_accuracy), fontsize=9, color="#d62728")

    ax.set_xlabel("avg cost per query")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs cost -- LTT-Router carries a risk guarantee none of the others report")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# CLI
def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LTT router experiments -> PNGs")
    parser.add_argument("--config", required=True, help="Baseline config YAML path")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--alphas", default="0.10,0.15,0.20,0.25,0.30,0.35",
                        help="comma-separated alphas for the sweep")
    parser.add_argument("--models", default="", help="comma-separated model subset")
    args = parser.parse_args(argv)

    from baselines.adaptors.ltt_adaptor import LTTAdaptor
    from baselines.ltt_router.routers.embedding_lr import CachingEmbedder

    subset = [m.strip() for m in args.models.split(",") if m.strip()] or None

    # Load the records ONCE; each trial re-splits with a different seed.
    base = LTTAdaptor(config_path=args.config, seed=0)
    records = base._load_records()
    if subset:
        records = [r for r in records if r.model_name in subset]

    # Embed every unique prompt ONCE and reuse across all
    # trials/α-values/Pareto-budget runs.
    print("[0/3] precomputing prompt embeddings (one pass) ...", flush=True)
    import time as _t
    _e0 = _t.time()
    embedder = CachingEmbedder()
    embedder.precompute(list({r.prompt for r in records}))
    print(f"      cached {len(embedder._cache)} unique prompts in {_t.time()-_e0:.1f}s",
          flush=True)

    def make_result(seed: int, apply_pareto: bool):
        ad = LTTAdaptor(config_path=args.config, seed=seed)
        return ad.run(alpha=args.alpha, delta=args.delta,
                      apply_pareto=apply_pareto, records=records, embed_fn=embedder)

    print(f"[1/3] repeated trials (n={args.n_trials}) ...", flush=True)
    out_par = run_repeated_trials(make_result, args.n_trials, apply_pareto=True, label="Pareto")
    out_bud = run_repeated_trials(make_result, args.n_trials, apply_pareto=False, label="budget")
    p1 = plot_guarantee_histogram(out_par, out_bud, args.alpha, args.delta, args.outdir)
    print(f"      -> {p1}")
    print(f"      +Pareto:     true risk {pooled_true_risk(out_par):.3f}  "
          f"corrected viol {corrected_violation_rate(out_par, args.alpha, args.delta):.1%}  "
          f"(raw {raw_violation_rate(out_par, args.alpha):.1%})")
    print(f"      budget-only: true risk {pooled_true_risk(out_bud):.3f}  "
          f"corrected viol {corrected_violation_rate(out_bud, args.alpha, args.delta):.1%}  "
          f"(raw {raw_violation_rate(out_bud, args.alpha):.1%})")

    print("[2/3] alpha-sweep ...", flush=True)
    alphas = [float(a) for a in args.alphas.split(",")]
    def make_at_alpha(a):
        ad = LTTAdaptor(config_path=args.config, seed=0)
        return ad.run(alpha=a, delta=args.delta, records=records, embed_fn=embedder)
    evals = run_alpha_sweep(make_at_alpha, alphas)
    p2 = plot_alpha_sweep(evals, alphas, args.outdir)
    print(f"      -> {p2}")

    print("[3/3] benchmark comparison ...", flush=True)
    ours = evaluate(make_result(0, True))
    p3 = plot_benchmark_comparison(ours, repo_root=".", outdir=args.outdir)
    print(f"      -> {p3}")


if __name__ == "__main__":
    main()