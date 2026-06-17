"""
Verification for ltt_split.

Tests the four properties that must hold if the split is correct:
  1. NO LEAKAGE: the three prompt-sets are pairwise disjoint.
  2. PROPORTIONS: record counts land near 60/20/20.
  3. REPRODUCIBILITY: same seed => byte-identical split.
  4. COVERAGE: every dataset appears in all three piles.
"""

from collections import defaultdict
from baselines import BaselineDataLoader
from baselines.ltt_v1.ltt_split import three_way_split


def prompt_set(records):
    return set(r.prompt for r in records)


def main():
    loader = BaselineDataLoader("config/baseline_config_performance_cost.yaml")
    records = loader.load_all_records()

    train, calib, test = three_way_split(
        records, train_frac=0.6, calib_frac=0.2, random_seed=42
    )

    tp, cp, sp = prompt_set(train), prompt_set(calib), prompt_set(test)

    # Check 1: NO LEAKAGE 
    leak_tc = tp & cp
    leak_ts = tp & sp
    leak_cs = cp & sp
    print("CHECK 1: LEAKAGE (must all be 0)")
    print(f"  train ∩ calib: {len(leak_tc)}")
    print(f"  train ∩ test:  {len(leak_ts)}")
    print(f"  calib ∩ test:  {len(leak_cs)}")
    no_leak = (len(leak_tc) == len(leak_ts) == len(leak_cs) == 0)
    print(f"  -> {'PASS' if no_leak else 'FAIL'}")

    # Check 2: PROPORTIONS
    total_recs = len(train) + len(calib) + len(test)
    total_prompts = len(tp) + len(cp) + len(sp)
    print("\nCHECK 2: PROPORTIONS (target 60/20/20)")
    print(f"  records: train {len(train)} ({len(train)/total_recs:.1%}) | "
          f"calib {len(calib)} ({len(calib)/total_recs:.1%}) | "
          f"test {len(test)} ({len(test)/total_recs:.1%})")
    print(f"  unique prompts: train {len(tp)} | calib {len(cp)} | test {len(sp)} "
          f"(total {total_prompts})")

    # Check 3: REPRODUCIBILITY
    train2, calib2, test2 = three_way_split(
        records, train_frac=0.6, calib_frac=0.2, random_seed=42
    )
    same = (prompt_set(train2) == tp and prompt_set(calib2) == cp
            and prompt_set(test2) == sp)
    print("\nCHECK 3: REPRODUCIBILITY (same seed => identical)")
    print(f"  -> {'PASS' if same else 'FAIL'}")

    # Check 4: COVERAGE (every dataset in all three piles) 
    def datasets_of(records):
        return set(r.dataset_id for r in records)
    all_ds = datasets_of(records)
    missing = []
    for name, recs in [("train", train), ("calib", calib), ("test", test)]:
        miss = all_ds - datasets_of(recs)
        if miss:
            missing.append((name, miss))
    print("\nCHECK 4: COVERAGE (every dataset in every pile)")
    if not missing:
        print(f"  all {len(all_ds)} datasets present in all three piles -> PASS")
    else:
        for name, miss in missing:
            print(f"  {name} missing: {miss}")
        print("  -> FAIL")

    # Per-dataset breakdown
    print("\nPER-DATASET PROMPT COUNTS (train/calib/test) ===")
    def ds_prompt_counts(records):
        d = defaultdict(set)
        for r in records:
            d[r.dataset_id].add(r.prompt)
        return {k: len(v) for k, v in d.items()}
    tc, cc, sc = ds_prompt_counts(train), ds_prompt_counts(calib), ds_prompt_counts(test)
    for ds in sorted(all_ds):
        print(f"  {ds:16s} {tc.get(ds,0):5d} / {cc.get(ds,0):4d} / {sc.get(ds,0):4d}")

    print("\n" + ("ALL CHECKS PASSED" if (no_leak and same and not missing)
                  else "SOME CHECKS FAILED — see above"))


if __name__ == "__main__":
    main()