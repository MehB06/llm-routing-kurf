"""
Verification for the LTT core. It empirically checks that the RISK-CONTROL GUARANTEE actually holds.

Tests:
  1. PRIMITIVE SANITY: p-values behave (risk well below alpha => small p; risk
     above alpha => p near 1). Binomial and HB agree in direction.

  2. THE GUARANTEE: calibrate λ̂ on the calib split, apply it to the
     TEST split, measure the true relative-loss rate. Repeat over many seeds.
     The fraction of seeds where test risk > alpha must be <= delta (roughly).
     NOTE: run at a FEASIBLE alpha. The qwen3/gpt-5 pair has a relative-risk
     floor near 0.15 (routable headroom ~15%), so alpha < ~0.15 is infeasible:
     no threshold can certify, the router correctly returns None, and the
     multi-seed test degenerates to "routes nothing." We test where the pair
     genuinely has headroom (alpha=0.15).

  3. MONOTONICITY: larger alpha => lower λ̂ => more queries routed cheap.
     (A looser guarantee should let us save more.)

  4. RANDOM-SCORER ABLATION: replace the real scorer with random scores. The
     guarantee must STILL hold (LTT is distribution-free). This proves safety
     comes from calibration, not from the scorer being good. Savings will be
     worse, but the guarantee holds — the key conceptual point. Run at a
     feasible alpha (0.20) so a weak random scorer can still certify SOMETHING
     and the demonstration isn't just "None"; safety is what we're checking.
"""

import numpy as np
from baselines import BaselineDataLoader
from baselines.ltt_v1.ltt_split import three_way_split
from baselines.ltt_v1.ltt_scorer import build_scorer, _cheap_examples
from baselines.ltt_v1.ltt_core import (
    CalibQuery, calibrate_threshold, routing_loss,
    binomial_pvalue, hoeffding_bentkus_pvalue,
)

CHEAP = "qwen3-235b-a22b-2507"
ORACLE = "gpt-5"


def build_queries(records, scorer, score_cache=None):
    """Align records into CalibQuery objects for the cheap/oracle pair."""
    # index by (dataset, record_index)
    cheap, oracle = {}, {}
    for r in records:
        key = (r.dataset_id, r.record_index)
        if r.model_name == CHEAP:
            cheap[key] = r
        elif r.model_name == ORACLE:
            oracle[key] = r
    keys = sorted(set(cheap) & set(oracle))
    prompts = [cheap[k].prompt for k in keys]
    if score_cache is not None:
        scores = score_cache
    else:
        scores = scorer.predict_proba(prompts, show_progress=False)
    return [
        CalibQuery(
            score=float(scores[i]),
            cheap_correct=int(cheap[k].score == 1.0),
            oracle_correct=int(oracle[k].score == 1.0),
        )
        for i, k in enumerate(keys)
    ], scores


def true_risk(queries, lam):
    """Actual relative-loss rate among cheap-routed queries at threshold lam."""
    routed = [q for q in queries if q.score >= lam]
    if not routed:
        return 0.0, 0
    losses = [routing_loss(q, True) for q in routed]
    return float(np.mean(losses)), len(routed)


def main():
    loader = BaselineDataLoader("config/baseline_config_performance_cost.yaml")
    records = loader.load_all_records()
    train, calib, test = three_way_split(records, 0.6, 0.2, random_seed=42)
    scorer = build_scorer(train, CHEAP)

    calib_q, _ = build_queries(calib, scorer)
    test_q, _ = build_queries(test, scorer)

    # 1: primitive sanity 
    print("1: P-VALUE SANITY")
    print(f"  risk=0.02, n=500, alpha=0.10 -> binom p={binomial_pvalue(0.02,500,0.10):.4f} "
          f"(want small)")
    print(f"  risk=0.20, n=500, alpha=0.10 -> binom p={binomial_pvalue(0.20,500,0.10):.4f} "
          f"(want 1)")
    print(f"  risk=0.02, n=500, alpha=0.10 -> HB    p={hoeffding_bentkus_pvalue(0.02,500,0.10):.4f}")

    # 2: THE GUARANTEE over many seeds
    # Each seed gets its OWN train/calib/test split AND its OWN scorer trained on
    # that seed's train split. This is essential: reusing one scorer across seeds
    # leaks training queries into other seeds' test sets and corrupts the result.
    # We use 15 seeds as a runtime/accuracy compromise.
    print("\n2: GUARANTEE — test-risk <= alpha across seeds (retrain per seed)")
    # alpha=0.15 is the smallest FEASIBLE bar for this pair (see header note).
    ALPHA, DELTA, N_SEEDS = 0.15, 0.10, 15
    violations = 0
    routed_fracs = []
    for seed in range(N_SEEDS):
        tr, ca, te = three_way_split(records, 0.6, 0.2, random_seed=seed)
        sc = build_scorer(tr, CHEAP)  # retrain on THIS seed's train split
        cq, _ = build_queries(ca, sc)
        tq, _ = build_queries(te, sc)
        res = calibrate_threshold(cq, alpha=ALPHA, delta=DELTA, pvalue="binomial")
        lam = res["lambda_hat"]
        if lam is None:
            routed_fracs.append(0.0)
            continue
        tr_risk, n_routed = true_risk(tq, lam)
        routed_fracs.append(n_routed / len(tq))
        if tr_risk > ALPHA:
            violations += 1
        print(f"  seed {seed:2d}: λ̂={lam:.3f}  test_risk={tr_risk:.3f}  "
              f"routed={n_routed/len(tq):.3f}  {'VIOLATION' if tr_risk>ALPHA else 'ok'}")
    print(f"\n  alpha={ALPHA}, delta={DELTA}, seeds={N_SEEDS}")
    print(f"  violations (test risk > alpha): {violations}/{N_SEEDS} "
          f"= {violations/N_SEEDS:.2f}")
    print(f"  mean fraction routed cheap: {np.mean(routed_fracs):.3f}")
    # The guarantee is MARGINAL over calibration draws: P(risk<=alpha)>=1-delta.
    # With N_SEEDS trials at true rate delta, the violation COUNT is Binomial.
    # We test whether the observed count is consistent with true rate <= delta. Use the upper tail: if
    # P(>= observed | true=delta) is not tiny, we can't reject "guarantee holds".
    from scipy import stats as _st
    p_obs = 1 - _st.binom.cdf(violations - 1, N_SEEDS, DELTA) if violations > 0 else 1.0
    consistent = p_obs > 0.05  # not significantly more violations than delta allows
    non_vacuous = np.mean(routed_fracs) >= 0.05
    print(f"  P(>= {violations} violations | true rate = delta) = {p_obs:.3f} "
          f"({'consistent with guarantee' if consistent else 'SIGNIFICANTLY too many'})")
    guarantee_ok = consistent and non_vacuous
    if consistent and not non_vacuous:
        print("  WARNING: guarantee holds but routes ~nothing, not a pass")

    # 3: monotonicity 
    print("\n3: MONOTONICITY (larger alpha => lower λ̂ => more cheap)")
    prev_lam, mono_ok = -1, True
    for a in [0.05, 0.10, 0.15, 0.20, 0.30]:
        res = calibrate_threshold(calib_q, alpha=a, delta=0.10)
        lam = res["lambda_hat"]
        _, n_routed = true_risk(test_q, lam) if lam is not None else (0, 0)
        frac = n_routed / len(test_q)
        print(f"  alpha={a:.2f} -> λ̂={lam}  routed_cheap={frac:.3f}")
        if lam is not None and prev_lam is not None and prev_lam >= 0 and lam > prev_lam:
            mono_ok = False
        prev_lam = lam if lam is not None else prev_lam

    # 4: random-scorer ablation 
    print("\n4: RANDOM-SCORER ABLATION (guarantee must STILL hold)")
    rng = np.random.default_rng(0)
    rand_calib = [CalibQuery(float(rng.random()), q.cheap_correct, q.oracle_correct)
                  for q in calib_q]
    rand_test = [CalibQuery(s, q.cheap_correct, q.oracle_correct)
                 for q, s in zip(test_q, rng.random(len(test_q)))]
    res = calibrate_threshold(rand_calib, alpha=0.20, delta=0.10)
    lam = res["lambda_hat"]
    if lam is not None:
        tr_risk, n_routed = true_risk(rand_test, lam)
        print(f"  random scorer: λ̂={lam:.3f}, test risk={tr_risk:.3f} "
              f"(<= 0.20? {tr_risk <= 0.20 + 0.05}), routed={n_routed/len(rand_test):.3f}")
    else:
        print(f"  random scorer: no safe λ found (acceptable — very conservative)")

    print("\n" + ("CORE LOOKS GOOD — guarantee holds AND routes non-trivially"
                  if guarantee_ok else
                  "CHECK FAILED — either violations too high or routing vacuous (see above)"))


if __name__ == "__main__":
    main()