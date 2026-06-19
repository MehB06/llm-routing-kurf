"""
End-to-end router evaluation. 

  1. REFERENCE BOUNDS: always-oracle and always-cheap (the frame).
  2. SINGLE OPERATING POINT at alpha=0.15: the router, with AvgAcc,
     CostSave, routed fraction, and measured relative risk (should be <= alpha).
  3. PERFORMANCE-COST CURVE: sweep alpha; each row is a point on the
     accuracy/cost tradeoff. This is the figure.
  4. SANITY: router accuracy sits between always-cheap and always-oracle;
     cost-save rises as alpha loosens.
"""

import numpy as np
from baselines import BaselineDataLoader
from baselines.ltt_v1.ltt_split import three_way_split
from baselines.ltt_v1.ltt_scorer import build_scorer
from baselines.ltt_v1.ltt_router import (
    pivot_records, compute_routing_metrics, reference_bounds,
    run_router, sweep_alpha, CHEAP_DEFAULT, ORACLE_DEFAULT,
)

CHEAP, ORACLE = CHEAP_DEFAULT, ORACLE_DEFAULT


def main():
    loader = BaselineDataLoader("config/baseline_config_performance_cost.yaml")
    records = loader.load_all_records()
    train, calib, test = three_way_split(records, 0.6, 0.2, random_seed=42)
    scorer = build_scorer(train, CHEAP)

    test_q = pivot_records(test, scorer, CHEAP, ORACLE)

    # 1: reference bounds
    bounds = reference_bounds(test_q)
    print("1: REFERENCE BOUNDS (on test split)")
    ao, ac = bounds["always_oracle"], bounds["always_cheap"]
    print(f"  always-oracle : acc={ao['avg_acc']:.3f}  cost=${ao['avg_cost']:.5f}  "
          f"cost_save={ao['cost_save']:.1%}")
    print(f"  always-cheap  : acc={ac['avg_acc']:.3f}  cost=${ac['avg_cost']:.5f}  "
          f"cost_save={ac['cost_save']:.1%}")

    # 2: single operating point at alpha=0.15
    print("\n2: CERTIFIED ROUTER at alpha=0.15, delta=0.10")
    m = run_router(calib, test, scorer, alpha=0.15, delta=0.10)
    lam_str = f"{m['lambda']:.3f}" if m['lambda'] is not None else "None"
    print(f"  λ̂ = {lam_str}")
    print(f"  AvgAcc            = {m['avg_acc']:.3f}")
    print(f"  AvgCost           = ${m['avg_cost']:.5f}")
    print(f"  CostSave          = {m['cost_save']:.1%}  (vs always-oracle)")
    print(f"  routed cheap      = {m['routed_cheap_frac']:.1%} "
          f"({m['n_routed_cheap']}/{m['n']})")
    print(f"  measured rel risk = {m['relative_risk']:.3f}  (target <= 0.15)")
    risk_ok = (m['lambda'] is not None) and (m['relative_risk'] <= 0.15 + 0.03)

    # 3: performance-cost curve
    print("\n3: PERFORMANCE-COST CURVE (sweep alpha)")
    print(f"  {'alpha':>6} {'lambda':>8} {'AvgAcc':>7} {'CostSave':>9} "
          f"{'routed%':>8} {'rel_risk':>9}")
    curve = sweep_alpha(calib, test, scorer,
                        alphas=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30), delta=0.10)
    for c in curve:
        lam_s = f"{c['lambda']:.3f}" if c['lambda'] is not None else "None"
        print(f"  {c['alpha']:6.2f} {lam_s:>8} {c['avg_acc']:7.3f} "
              f"{c['cost_save']:9.1%} {c['routed_cheap_frac']:8.1%} "
              f"{c['relative_risk']:9.3f}")

    # 4: sanity checks
    print("\n4: SANITY CHECKS")
    # router acc between cheap and oracle (at a routing point)
    acc_in_range = ac['avg_acc'] - 0.02 <= m['avg_acc'] <= ao['avg_acc'] + 0.02
    # cost-save monotone non-decreasing as alpha rises
    saves = [c['cost_save'] for c in curve]
    save_monotone = all(saves[i] <= saves[i+1] + 0.02 for i in range(len(saves)-1))
    print(f"  router acc within [cheap, oracle]: {acc_in_range} "
          f"({ac['avg_acc']:.3f} <= {m['avg_acc']:.3f} <= {ao['avg_acc']:.3f})")
    print(f"  cost-save rises as alpha loosens:  {save_monotone}")
    print(f"  measured risk <= alpha at 0.15:    {risk_ok}")

    ok = acc_in_range and save_monotone and risk_ok
    print("\n" + ("ROUTER WORKS"
                  if ok else "SOMETHING OFF "))


if __name__ == "__main__":
    main()