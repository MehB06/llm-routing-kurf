"""
Evaluation for the LTT scorer.

A scorer is good BY DEGREE,so we MEASURE rather than assert. 
We train on the train split, then evaluate on the CALIBRATION 
split (held-out).

Metrics:
  1. AUC  — given a query cheap got right and one cheap got wrong, how often does
            the scorer rank the right one higher? 0.5 = chance, 1.0 = perfect.
            This is number for scorer quality.
  2. Brier score / calibration — do predicted probabilities match reality?
            Queries scored ~0.7 should succeed ~70% of the time. 
            Matters because LTT thresholds directly on these scores.
  3. Not-broken checks — outputs in [0,1], genuine spread (not all ~0.5),
            and AUC beats a shuffled-label baseline.
"""

import numpy as np
from baselines import BaselineDataLoader
from baselines.ltt_v1.ltt_split import three_way_split
from baselines.ltt_v1.ltt_scorer import build_scorer, _cheap_examples

CHEAP = "qwen3-235b-a22b-2507"


def auc(labels, scores):
    """ROC AUC via the rank-sum (Mann-Whitney) identity."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    # (small correction; fine for a sanity check)
    sum_ranks_pos = ranks[labels == 1].sum()
    return (sum_ranks_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    loader = BaselineDataLoader("config/baseline_config_performance_cost.yaml")
    records = loader.load_all_records()
    train, calib, _test = three_way_split(records, 0.6, 0.2, random_seed=42)

    # Train scorer on train split
    scorer = build_scorer(train, CHEAP)

    # Evaluate on calibration split (held-out)
    calib_prompts, calib_labels = _cheap_examples(calib, CHEAP)
    calib_scores = scorer.predict_proba(calib_prompts)

    # 1: AUC 
    a = auc(calib_labels, calib_scores)
    print("\n1: AUC on held-out calibration split ")
    print(f"AUC = {a:.3f}")

    # 2: Calibration (reliability by bins) + Brier 
    print("\n2: CALIBRATION (predicted vs actual success by score bin)")
    bins = np.linspace(0, 1, 6)  # 5 bins
    idx = np.digitize(calib_scores, bins) - 1
    idx = np.clip(idx, 0, 4)
    print(f"  {'bin':>10}  {'n':>6}  {'mean_pred':>10}  {'actual':>8}")
    for b in range(5):
        m = idx == b
        if m.sum() == 0:
            continue
        print(f"  {bins[b]:.1f}-{bins[b+1]:.1f}   {m.sum():6d}  "
              f"{calib_scores[m].mean():10.3f}  {calib_labels[m].mean():8.3f}")
    brier = np.mean((calib_scores - calib_labels) ** 2)
    print(f"  Brier score = {brier:.3f}  (lower is better; 0.25 = uninformative)")

    # 3: Not-broken checks 
    print("\n3: SANITY CHECKS")
    in_range = (calib_scores.min() >= 0) and (calib_scores.max() <= 1)
    spread = calib_scores.std()
    rng = np.random.default_rng(0)
    shuffled_auc = auc(rng.permutation(calib_labels), calib_scores)
    print(f"  outputs in [0,1]:        {in_range}  "
          f"(min {calib_scores.min():.3f}, max {calib_scores.max():.3f})")
    print(f"  score spread (std):      {spread:.3f}  (near 0 => degenerate)")
    print(f"  shuffled-label AUC:      {shuffled_auc:.3f}  (should be ~0.5)")
    print(f"  real AUC beats shuffled: {a > shuffled_auc + 0.05}")

    # Verdict 
    good = (a >= 0.60) and in_range and (spread > 0.05) and (a > shuffled_auc + 0.05)
    print("\n" + ("SCORER GOOD"
                  if good else
                  "SCORER WEAK/BROKEN"))
    print(f"(headline: AUC={a:.3f}, Brier={brier:.3f}, spread={spread:.3f})")


if __name__ == "__main__":
    main()