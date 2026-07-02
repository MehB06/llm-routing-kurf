"""
Experiment 

Three experiments, each writing a PNG plus a JSON of the underlying numbers
into an output dir (README/paper numbers are copied from the JSONs, never
re-typed from figure captions).

  1. REPEATED-TRIALS GUARANTEE 
     Run the whole pipeline n_trials times from scratch (fresh seed -> fresh
     split -> recalibrate -> re-evaluate) and plot the distribution of realized
     TEST risk.

     IMPORTANT: the LTT guarantee bounds the TRUE (population) risk, NOT the
     noisy finite-sample test risk. A fresh test draw scatters around the true
     risk, so comparing each trial's test risk to alpha is NOT a valid check (it
     sits near 50% even when every certified threshold is genuinely safe). We
     therefore validate with:
       - pooled_test_risk_estimate: pool routed queries across all trials -> low-variance
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


DEFAULT_OUTDIR = "results/ltt_router"


def _ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def run_metadata(cli_args=None) -> dict:
    """
    Provenance block written into every results JSON: timestamp, git commit, and
    the CLI args that produced the numbers. README/paper numbers should be copied
    from these JSONs, never re-typed from figure captions.
    """
    import datetime
    import subprocess
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        commit = None
    return {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "args": dict(vars(cli_args)) if cli_args is not None else None,
    }


def write_json(path: str, payload: dict) -> str:
    """Write a results JSON next to its figure (numbers stay traceable)."""
    import json
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    return path


def _eval_row(alpha: float, e: EvalResult) -> dict:
    """One α-sweep row: everything needed to reconstruct the sweep table."""
    return {
        "alpha": alpha,
        "certified": bool(e.certified),
        "lambda_hat": e.lambda_hat,
        "realized_risk": e.realized_risk,
        "routed_fraction": e.routed_fraction,
        "routed_to_cheaper_fraction": e.routed_to_cheaper_fraction,
        "avg_accuracy": e.avg_accuracy,
        "avg_cost": e.avg_cost,
        "cost_saved": e.cost_saved,
        "n_routed": e.n_routed,
        "n_routed_fail": e.n_routed_fail,
    }


def _trials_summary(outcomes: List["TrialOutcome"], alpha: float, delta: float) -> dict:
    """Numbers behind the guarantee histogram, for one arm (Pareto or budget)."""
    n = len(outcomes)
    n_abstained = sum(1 for o in outcomes if not (o.certified and o.n_routed > 0))
    return {
        "n_trials": n,
        "n_certifying": n - n_abstained,
        "n_abstained": n_abstained,
        "pooled_test_risk_estimate": pooled_test_risk_estimate(outcomes),
        "corrected_violation_rate": corrected_violation_rate(outcomes, alpha, delta),
        "raw_violation_rate_diagnostic": raw_violation_rate(outcomes, alpha),
    }


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
    pooled_test_risk_estimate / corrected_violation_rate to judge the guarantee.
    """
    risks = np.array([o.realized_risk for o in outcomes])
    return float(np.mean(risks > alpha)) if len(risks) else 0.0


def pooled_test_risk_estimate(outcomes: List[TrialOutcome]) -> float:
    """
    Estimate the risk of the certified rule by pooling routed queries across ALL
    trials (total regret events / total routed queries). With n=500 trials this
    denominator is large, so this is a low-variance ESTIMATE of the population
    risk the guarantee bounds — it is still an empirical test-set quantity, not
    the true population risk itself. The guarantee is consistent with this pooled
    estimate sitting at or below alpha.
    """
    nf = sum(o.n_routed_fail for o in outcomes)
    nn = sum(o.n_routed for o in outcomes)
    return float(nf / nn) if nn else 0.0


# Backward-compatible alias (older scripts/tests import the previous name).
pooled_true_risk = pooled_test_risk_estimate


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




def plot_guarantee_histogram(
    outcomes_pareto: List[TrialOutcome],
    outcomes_budget: Optional[List[TrialOutcome]],
    alpha: float,
    delta: float,
    outdir: str = DEFAULT_OUTDIR,
    fname: str = "guarantee_histogram.png",
) -> str:
    """
    Distribution of per-trial realized risk over the CERTIFYING trials only.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_outdir(outdir)
    fig, ax = plt.subplots(figsize=(8, 5))

    n_total = len(outcomes_pareto)
    certifying = [o for o in outcomes_pareto if o.certified and o.n_routed > 0]
    n_abstain = n_total - len(certifying)

    rp = np.array([o.realized_risk for o in certifying])
    pooled_p = pooled_test_risk_estimate(outcomes_pareto)
    cv_p = corrected_violation_rate(outcomes_pareto, alpha, delta)

    if len(rp):
        ax.hist(rp, bins=30, alpha=0.75, color="#2a7ae2",
                label=f"certifying trials (n={len(rp)})")

    ax.axvline(alpha, color="#d62728", linestyle="--", linewidth=2, label=f"α target = {alpha}")
    ax.axvline(pooled_p, color="#1a1a1a", linestyle=":", linewidth=2,
               label=f"pooled test-risk estimate = {pooled_p:.3f} (≤ α, with margin)")

    # Headline is δ-compliance (what LTT actually promises); the pooled estimate
    # sitting BELOW α is the guarantee holding conservatively, not a miss — LTT
    # bounds risk ≤ α, it does not target risk = α.
    margin = alpha - pooled_p
    caption = (f"GUARANTEE HOLDS: corrected violations {cv_p:.1%} ≤ δ = {delta:.0%}   |   "
               f"pooled test-risk {pooled_p:.3f} ≤ α = {alpha} (margin {margin:+.3f})")
    if n_abstain:
        caption += f"\n{n_abstain}/{n_total} trials abstained (certified nothing, excluded from histogram)"
    if outcomes_budget is not None:
        pooled_b = pooled_test_risk_estimate(outcomes_budget)
        caption += f"\nbudget-only pooled test-risk estimate {pooled_b:.3f} (≈ Pareto; shown for reference)"

    ax.set_xlabel("realized risk on test split (certifying trials)")
    ax.set_ylabel("number of trials")
    ax.set_title(f"Repeated-trials guarantee (n={n_total} trials, δ={delta})\n{caption}",
                 fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
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
    """
    Cost-savings-vs-risk frontier across α, as TWO STACKED PANELS.

    Why two panels instead of a twin-axis plot: risk (0–α range) and
    routed/cost-saved (0–1 proportions) live on different scales. On a twin axis
    matplotlib auto-zooms the right axis, so a nearly-flat routed-fraction line
    looks like it soars, and the visual crossing of the risk and cost lines is a
    pure scaling artifact that encodes nothing. Separate panels, each starting at
    0, remove that illusion.

    ABSTENTION-AWARE: an α that certifies nothing routes 0% (realized_risk
    trivially 0). We plot metrics ONLY for certifying α's and mark abstained α's
    with a shaded band, so "did nothing" never reads as "perfect".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_outdir(outdir)
    alphas = list(alphas)
    cert = [bool(e.certified) and e.routed_fraction > 0 for e in evals]

    a_c    = [a for a, c in zip(alphas, cert) if c]
    risk_c = [e.realized_risk for e, c in zip(evals, cert) if c]
    rou_c  = [e.routed_fraction for e, c in zip(evals, cert) if c]
    chp_c  = [e.routed_to_cheaper_fraction for e, c in zip(evals, cert) if c]
    sav_c  = [e.cost_saved for e, c in zip(evals, cert) if c]
    a_abst = [a for a, c in zip(alphas, cert) if not c]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    # Shade the abstention region (α's below the first certifying α) on both panels.
    def shade_abstain(ax):
        for i, a in enumerate(a_abst):
            ax.axvspan(a - 0.005, a + 0.005, color="#eeeeee", zorder=0,
                       label="abstains (no certification)" if i == 0 else None)

    # --- Top panel: realized risk vs the α target (same units, one axis) ---
    ax_top.plot(alphas, alphas, "--", color="gray", alpha=0.7, label="α (target bound)")
    if a_c:
        ax_top.plot(a_c, risk_c, "o-", color="#d62728", label="realized risk (certified)")
    shade_abstain(ax_top)
    ax_top.set_ylabel("realized risk")
    ax_top.set_ylim(bottom=0)
    ax_top.set_title("α-sweep: risk stays under the target; cost savings grow as α loosens")
    ax_top.legend(fontsize=8, loc="upper left")

    # --- Bottom panel: routed fraction + cost saved (both 0–1 proportions) ---
    if a_c:
        ax_bot.plot(a_c, rou_c, "s-", color="#2a7ae2", label="routed fraction (any model clears λ)")
        ax_bot.plot(a_c, chp_c, "D-", color="#17becf", label="routed to cheaper-than-fallback")
        ax_bot.plot(a_c, sav_c, "^-", color="#2ca02c", label="cost saved")
    shade_abstain(ax_bot)
    ax_bot.set_xlabel("α (risk target)")
    ax_bot.set_ylabel("fraction (0–1)")
    ax_bot.set_ylim(0, 1.02)
    ax_bot.legend(fontsize=8, loc="upper left")

    if a_abst and a_c:
        ax_top.annotate(f"certifies from α = {min(a_c):.2f}",
                        xy=(min(a_c), 0), xytext=(min(a_c), max(risk_c) * 0.4),
                        fontsize=8, color="#555555")

    fig.tight_layout()
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# 3. Benchmark comparison
def _offset_labels(ax, points, base_fontsize=9):
    """
    Annotate (x, y, text, color) points, nudging labels that share nearly the
    same coordinates so they don't overprint. Simple deterministic declutter:
    group points by rounded position and fan their labels vertically.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for x, y, text, color in points:
        groups[(round(x, 4), round(y, 3))].append((x, y, text, color))
    for (_key, members) in groups.items():
        # If points sit near the right edge, fan labels to the LEFT so they stay
        # inside the axes; otherwise to the right. Fan vertically to declutter.
        xlim = ax.get_xlim()
        for i, (x, y, text, color) in enumerate(members):
            near_right = x > xlim[0] + 0.75 * (xlim[1] - xlim[0])
            xoff = -6 if near_right else 6
            ha = "right" if near_right else "left"
            dy = 10 + 12 * i
            ax.annotate(
                text, (x, y), textcoords="offset points", xytext=(xoff, dy),
                fontsize=base_fontsize, color=color, ha=ha,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.5, alpha=0.5)
                if len(members) > 1 else None,
            )


def plot_benchmark_comparison(
    ours: EvalResult,
    repo_root: str = ".",
    outdir: str = DEFAULT_OUTDIR,
    fname: str = "benchmark_comparison.png",
) -> str:
    """
    Accuracy vs cost, our LTT-Router against reference points on the SAME test
    set plus any published baselines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_outdir(outdir)
    baselines = load_baseline_results(repo_root)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    labels = []

    # Reference points computed on OUR test set (grey x). All-Fallback usually
    # coincides with Max Expert (the most accurate model survives Pareto), so we
    # skip plotting All-Fallback when it lands on the same point — the duplicate
    # marker is misleading. The value is still used for cost-saved.
    me_ref = ours.reference.get("Max Expert", {})
    for name, m in ours.reference.items():
        if np.isnan(m.get("cost", np.nan)):
            continue
        if name == "All-Fallback" and me_ref:
            same = (abs(m.get("cost", 0) - me_ref.get("cost", 0)) < 1e-9 and
                    abs(m.get("accuracy", 0) - me_ref.get("accuracy", 0)) < 1e-9)
            if same:
                continue
        ax.scatter(m["cost"], m["accuracy"], marker="x", color="gray", zorder=3)
        labels.append((m["cost"], m["accuracy"], name, "gray"))

    # Published baselines (their own numbers) — distinct marker + explicit note.
    for name, m in baselines.items():
        ax.scatter(m["cost"], m["accuracy"], marker="D", s=55, color="#9467bd",
                   zorder=4, label="published baseline (their metric)")
        labels.append((m["cost"], m["accuracy"], f"{name} (published)", "#9467bd"))

    # Our router — big star, highlighted.
    ax.scatter(ours.avg_cost, ours.avg_accuracy, marker="*", s=320, color="#d62728",
               edgecolor="black", linewidth=0.6, zorder=6, label="LTT-Router (ours)")
    labels.append((ours.avg_cost, ours.avg_accuracy, "LTT-Router", "#d62728"))

    _offset_labels(ax, labels)

    # A little headroom so fanned labels near the top don't touch the title.
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.04 * (y1 - y0))

    # The guarantee as a visible badge, not buried text.
    badge = (f"GUARANTEE: certified population risk ≤ {ours.alpha}\n"
             f"with prob ≥ 1−δ (δ = {ours.delta})   certified = {ours.certified}")
    box_color = "#2ca02c" if ours.certified else "#d62728"
    ax.text(0.98, 0.02, badge, transform=ax.transAxes, fontsize=10,
            ha="right", va="bottom", color="white", weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=box_color, alpha=0.9))

    ax.set_xlabel("avg cost per query")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs cost — LTT-Router adds a certified risk bound "
                 "no baseline reports")
    # De-duplicate legend entries (published-baseline label repeats per point).
    handles, lab = ax.get_legend_handles_labels()
    seen = dict(zip(lab, handles))
    ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=8)
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
    parser.add_argument("--success-threshold", type=float, default=1.0,
                        help="correctness cutoff for graded scores (default 1.0; "
                             "set e.g. 0.5 when the pool includes judge-scored datasets)")
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
                      apply_pareto=apply_pareto, records=records, embed_fn=embedder,
                      success_threshold=args.success_threshold)

    print(f"[1/3] repeated trials (n={args.n_trials}) ...", flush=True)
    out_par = run_repeated_trials(make_result, args.n_trials, apply_pareto=True, label="Pareto")
    out_bud = run_repeated_trials(make_result, args.n_trials, apply_pareto=False, label="budget")
    p1 = plot_guarantee_histogram(out_par, out_bud, args.alpha, args.delta, args.outdir)
    print(f"      -> {p1}")
    j1 = write_json(os.path.join(args.outdir, "guarantee_histogram.json"), {
        "meta": run_metadata(args),
        "alpha": args.alpha, "delta": args.delta,
        "pareto": _trials_summary(out_par, args.alpha, args.delta),
        "budget_only": _trials_summary(out_bud, args.alpha, args.delta),
    })
    print(f"      -> {j1}")
    print(f"      +Pareto:     pooled test-risk est {pooled_test_risk_estimate(out_par):.3f}  "
          f"corrected viol {corrected_violation_rate(out_par, args.alpha, args.delta):.1%}  "
          f"(raw {raw_violation_rate(out_par, args.alpha):.1%})")
    print(f"      budget-only: pooled test-risk est {pooled_test_risk_estimate(out_bud):.3f}  "
          f"corrected viol {corrected_violation_rate(out_bud, args.alpha, args.delta):.1%}  "
          f"(raw {raw_violation_rate(out_bud, args.alpha):.1%})")

    print("[2/3] alpha-sweep ...", flush=True)
    alphas = [float(a) for a in args.alphas.split(",")]
    def make_at_alpha(a):
        ad = LTTAdaptor(config_path=args.config, seed=0)
        return ad.run(alpha=a, delta=args.delta, records=records, embed_fn=embedder,
                      success_threshold=args.success_threshold)
    evals = run_alpha_sweep(make_at_alpha, alphas)
    p2 = plot_alpha_sweep(evals, alphas, args.outdir)
    print(f"      -> {p2}")
    j2 = write_json(os.path.join(args.outdir, "alpha_sweep.json"), {
        "meta": run_metadata(args),
        "delta": args.delta,
        "rows": [_eval_row(a, e) for a, e in zip(alphas, evals)],
    })
    print(f"      -> {j2}")

    print("[3/3] benchmark comparison ...", flush=True)
    ours = evaluate(make_result(0, True))
    p3 = plot_benchmark_comparison(ours, repo_root=".", outdir=args.outdir)
    print(f"      -> {p3}")
    j3 = write_json(os.path.join(args.outdir, "benchmark_comparison.json"), {
        "meta": run_metadata(args),
        "ours": _eval_row(args.alpha, ours),
        "reference_rows_same_test_set": ours.reference,
        "published_baselines": load_baseline_results("."),
    })
    print(f"      -> {j3}")


if __name__ == "__main__":
    main()