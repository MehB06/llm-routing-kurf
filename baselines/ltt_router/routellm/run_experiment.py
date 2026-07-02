"""
Single-seed RouteLLM experiment runner (seed 42).

Why single-seed: the MF checkpoint is trained ONCE on seed-42 train_fit, and the
calibrators on seed-42 train_cal. The multi-seed guarantee histogram re-splits
per seed, which would leak seed-42 train prompts into other seeds' CALIB/TEST
and break disjointness. At seed 42 everything is disjoint by construction:
    train_fit -> MF | train_cal -> calibrators | CALIB -> LTT | TEST -> eval.

So for RouteLLM we report:
  1. an α-SWEEP (the certification frontier: where the calibrated MF scorer
     starts certifying) — printed as a table and written to JSON. With only a
     handful of certifying α's the sweep makes a weak FIGURE, so no PNG is
     generated for it; the numbers live in alpha_sweep_routellm.json.
  2. the BENCHMARK point at a chosen α — figure + JSON.
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import List, Optional

from baselines.ltt_router.eval.metrics import evaluate, load_baseline_results
from baselines.ltt_router.experiment import (
    run_alpha_sweep, plot_benchmark_comparison,
    run_metadata, write_json, _eval_row,
)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="RouteLLM MF experiment (single-seed)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--mf-checkpoint", required=True)
    ap.add_argument("--mf-calibrators", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.20,
                    help="α for the single benchmark point")
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--alphas", default="0.10,0.15,0.20,0.25,0.30,0.35")
    ap.add_argument("--outdir", default="results/ltt_router")
    ap.add_argument("--success-threshold", type=float, default=1.0,
                    help="correctness cutoff for graded scores (default 1.0; "
                         "set e.g. 0.5 when the pool includes judge-scored datasets)")
    args = ap.parse_args(argv)

    from baselines.adaptors.ltt_adaptor import LTTAdaptor
    from baselines.ltt_router.routers.embedding_lr import CachingEmbedder

    cals = pickle.load(open(args.mf_calibrators, "rb"))
    method = cals.get("method", "?")
    calmap = cals["calibrators"]
    print(f"[0/2] loaded MF checkpoint + {len(calmap)} calibrators (method={method})")

    base = LTTAdaptor(config_path=args.config, seed=args.seed)
    records = base._load_records()

    embedder = CachingEmbedder()
    embedder.precompute(list({r.prompt for r in records}))
    print(f"      cached {len(embedder._cache)} unique prompts")

    def run_at_alpha(a):
        ad = LTTAdaptor(config_path=args.config, seed=args.seed)
        return ad.run(alpha=a, delta=args.delta, records=records, embed_fn=embedder,
                      scorer_kind="routellm_mf",
                      mf_checkpoint=args.mf_checkpoint, mf_calibrators=calmap,
                      success_threshold=args.success_threshold)

    # 1. α-sweep: the certification frontier (table + JSON, no figure — see
    #    module docstring).
    print("[1/2] α-sweep (certification frontier) ...")
    alphas = [float(a) for a in args.alphas.split(",")]
    evals = run_alpha_sweep(run_at_alpha, alphas)
    print(f"\n  {'alpha':>6} {'certified':>10} {'routed':>8} {'risk':>8} "
          f"{'acc':>7} {'cost':>9} {'saved':>7}")
    for a, e in zip(alphas, evals):
        print(f"  {a:6.2f} {str(e.certified):>10} {e.routed_fraction:8.2%} "
              f"{e.realized_risk:8.3f} {e.avg_accuracy:7.3f} {e.avg_cost:9.4f} "
              f"{e.cost_saved:7.2%}")
    j_sweep = write_json(os.path.join(args.outdir, "alpha_sweep_routellm.json"), {
        "meta": run_metadata(args),
        "delta": args.delta,
        "calibration_method": method,
        "rows": [_eval_row(a, e) for a, e in zip(alphas, evals)],
    })
    print(f"  -> {j_sweep}")

    # 2. benchmark point — use the FIRST α that actually certifies, so we never
    #    plot an abstaining (certified=False) router as if it were a result.
    cert_alphas_all = [a for a, e in zip(alphas, evals) if e.certified and e.routed_fraction > 0]
    bench_alpha = next((a for a in cert_alphas_all if a >= args.alpha),
                       (min(cert_alphas_all) if cert_alphas_all else args.alpha))
    if bench_alpha != args.alpha:
        print(f"\n[2/2] requested α={args.alpha} did not certify; "
              f"plotting first certifying α={bench_alpha} instead")
    else:
        print(f"\n[2/2] benchmark point at α={bench_alpha} ...")
    ours = evaluate(run_at_alpha(bench_alpha))
    print(f"  certified={ours.certified}  routed={ours.routed_fraction:.2%}  "
          f"acc={ours.avg_accuracy:.3f}  cost={ours.avg_cost:.4f}  "
          f"risk={ours.realized_risk:.3f}")
    p_bench = plot_benchmark_comparison(ours, repo_root=".", outdir=args.outdir,
                                        fname="benchmark_comparison_routellm.png")
    print(f"  -> {p_bench}")
    j_bench = write_json(os.path.join(args.outdir, "benchmark_comparison_routellm.json"), {
        "meta": run_metadata(args),
        "ours": _eval_row(bench_alpha, ours),
        "reference_rows_same_test_set": ours.reference,
        "published_baselines": load_baseline_results("."),
        "calibration_method": method,
    })
    print(f"  -> {j_bench}")

    # First α (if any) at which the scorer certifies — the headline number.
    cert_alphas = [a for a, e in zip(alphas, evals) if e.certified]
    if cert_alphas:
        print(f"\nHEADLINE: calibrated RouteLLM-MF scorer first certifies at "
              f"α = {min(cert_alphas):.2f} under the LTT guarantee.")
    else:
        print("\nHEADLINE: no α in the sweep certified; widen --alphas upward.")


if __name__ == "__main__":
    main()