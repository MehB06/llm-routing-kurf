"""
LTT adaptor 

    load -> three-way split (train/calib/test)      
         -> train per-model scorer on TRAIN         
         -> pivot calib+test groups to QueryRecords 
         -> Router.fit on CALIB (Pareto + LTT)      
         -> route TEST QueryRecords                 
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from baselines.ltt_router.protocols import ModelSpec, QueryRecord
from baselines.ltt_router.splitting import three_way_split
from baselines.ltt_router.routers.embedding_lr import build_embedding_lr_router, EmbedFn
from baselines.ltt_router.routers.routellm_mf import build_routellm_mf_router
from baselines.ltt_router.core.routing import Router, RouterPlan
from baselines.ltt_router.core.calibration import MIN_ROUTED_DEFAULT


def build_model_specs(records: Sequence) -> List[ModelSpec]:
    """
    Derive the N-model universe: name-sorted unique models (stable indices), each
    with cost = mean per-record cost across its rows. Sorting by name gives a
    deterministic, reproducible index map.
    """
    names = sorted({r.model_name for r in records})
    cost_sums: Dict[str, float] = defaultdict(float)
    cost_counts: Dict[str, int] = defaultdict(int)
    for r in records:
        cost_sums[r.model_name] += float(r.cost)
        cost_counts[r.model_name] += 1
    specs = []
    for idx, name in enumerate(names):
        mean_cost = cost_sums[name] / cost_counts[name] if cost_counts[name] else 0.0
        specs.append(ModelSpec(name=name, cost=mean_cost, index=idx))
    return specs


def pivot_to_query_records(
    records: Sequence,
    models: List[ModelSpec],
    scorer,
) -> List[QueryRecord]:
    """
    Group by (dataset_id, prompt); for each group build the N-length arrays with
    evaluated[i] True only for models that actually have a row. Scores come from
    the injected scorer (batched over all prompts for efficiency).
    """
    name_to_index = {m.name: m.index for m in models}
    n_models = len(models)
    cost_by_index = np.array([m.cost for m in models], dtype=float)

    groups: Dict[Tuple[str, str], list] = defaultdict(list)
    for r in records:
        groups[(r.dataset_id, r.prompt)].append(r)

    keys = list(groups.keys())
    prompts = [prompt for (_ds, prompt) in keys]
    score_matrix = scorer.score_batch(prompts)   # [G, N]

    query_records: List[QueryRecord] = []
    for gi, (ds_id, prompt) in enumerate(keys):
        group = groups[(ds_id, prompt)]
        correct = np.zeros(n_models, dtype=int)
        evaluated = np.zeros(n_models, dtype=bool)
        cost = cost_by_index.copy()              # default to model's mean cost
        for r in group:
            idx = name_to_index.get(r.model_name)
            if idx is None:
                continue                          # model not in universe (shouldn't happen)
            evaluated[idx] = True
            correct[idx] = int(r.score == 1.0)
            cost[idx] = float(r.cost)             # prefer the real per-query cost
        if not evaluated.any():
            continue                              # skip empty groups
        query_records.append(QueryRecord(
            scores=score_matrix[gi],
            correct=correct,
            cost=cost,
            evaluated=evaluated,
            dataset_id=ds_id,
            prompt=prompt,
        ))
    return query_records


@dataclass
class AdaptorResult:
    """Compute both metric blocks."""
    plan: RouterPlan
    test_queries: List[QueryRecord]
    chosen_indices: np.ndarray            # int[M], router's choice per test query
    models: List[ModelSpec]
    alpha: float
    delta: float
    apply_pareto: bool
    seed: int


class LTTAdaptor:
    """
    Orchestrates the full v2 pipeline on benchmark data.

        adaptor = LTTAdaptor("config/baseline_config_performance_cost.yaml")
        result = adaptor.run(alpha=0.15)   
    """

    def __init__(
        self,
        config_path: str,
        train_frac: float = 0.6,
        calib_frac: float = 0.2,
        seed: int = 42,
    ):
        self.config_path = config_path
        self.train_frac = train_frac
        self.calib_frac = calib_frac
        self.seed = seed

    def run(
        self,
        alpha: float,
        delta: float = 0.10,
        apply_pareto: bool = True,
        embed_fn: Optional[EmbedFn] = None,
        scorer_kind: str = "embedding_lr",
        mf_checkpoint: Optional[str] = None,
        mf_calibrators: Optional[dict] = None,
        models_subset: Optional[List[str]] = None,
        min_routed: int = MIN_ROUTED_DEFAULT,
        records: Optional[Sequence] = None,
    ) -> AdaptorResult:
        """
        Load -> split -> train scorer -> calibrate -> route the test set.

        apply_pareto:   True = Pareto-filtered; False = budget-only ablation.
        embed_fn:       inject a stub in tests to skip the model download.
        models_subset:  restrict to these model names. None = all models in the data.
        min_routed:     power floor for FST eligibility (see calibration.py).
        records:        pre-loaded records (tests). None = load from benchmark.
        """
        if records is None:
            records = self._load_records()

        if models_subset is not None:
            keep = set(models_subset)
            records = [r for r in records if r.model_name in keep]

        train, calib, test = three_way_split(
            records, self.train_frac, self.calib_frac, self.seed
        )

        models = build_model_specs(records)

        # Train the scorer on TRAIN only.
        if scorer_kind == "routellm_mf":
            if mf_checkpoint is None or mf_calibrators is None:
                raise ValueError("routellm_mf requires mf_checkpoint and mf_calibrators")
            scorer = build_routellm_mf_router(models, mf_checkpoint, mf_calibrators, embed_fn=embed_fn)
        else:
            scorer = build_embedding_lr_router(train, models, embed_fn=embed_fn)

        # Pivot calib + test to QueryRecords (scores from the trained scorer).
        calib_queries = pivot_to_query_records(calib, models, scorer)
        test_queries = pivot_to_query_records(test, models, scorer)

        # Calibrate on CALIB (Pareto + cost-order + LTT).
        router = Router(scorer)
        plan = router.fit(
            calib_queries, alpha=alpha, delta=delta,
            apply_pareto=apply_pareto, min_routed=min_routed,
        )

        # Route the held-out TEST queries.
        chosen = router.route_batch(test_queries)

        return AdaptorResult(
            plan=plan,
            test_queries=test_queries,
            chosen_indices=chosen,
            models=models,
            alpha=alpha,
            delta=delta,
            apply_pareto=apply_pareto,
            seed=self.seed,
        )

    def _load_records(self):
        from baselines.data_loader import BaselineDataLoader
        loader = BaselineDataLoader(self.config_path)
        return loader.load_all_records()


# CLI (standard adaptor arguments)
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="LTT router benchmark adaptor")
    parser.add_argument("--config", required=True, help="Baseline config YAML path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--calib-frac", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.15, help="LTT risk target")
    parser.add_argument("--delta", type=float, default=0.10, help="LTT failure prob")
    parser.add_argument("--no-pareto", action="store_true", help="budget-only ablation")
    parser.add_argument("--models", default="", help="comma-separated model subset")
    parser.add_argument("--verbose", action="store_true",
                        help="print the calibration loss table (risk/n/p per λ) for diagnosis")
    args = parser.parse_args(argv)

    subset = [m.strip() for m in args.models.split(",") if m.strip()] or None

    adaptor = LTTAdaptor(
        config_path=args.config,
        train_frac=args.train_frac,
        calib_frac=args.calib_frac,
        seed=args.seed,
    )
    result = adaptor.run(
        alpha=args.alpha,
        delta=args.delta,
        apply_pareto=not args.no_pareto,
        models_subset=subset,
    )

    plan = result.plan
    print(f"models:           {[m.name for m in result.models]}")
    print(f"certified:        {plan.certified}")
    print(f"lambda_hat:       {plan.lambda_hat}")
    print(f"survivors:        {[result.models[i].name for i in plan.survivors]}")
    print(f"test queries:     {len(result.test_queries)}")
    routed_frac = float(np.mean(result.chosen_indices != plan.fallback_idx))
    print(f"routed non-fallback fraction: {routed_frac:.3f}")

    if args.verbose:
        # Show the calibration loss table so you can SEE whether alpha is
        # achievable.
        cal = plan.calibration
        print(f"\n  {'lambda':>7} {'n_routed':>9} {'risk':>7} {'p-value':>9} {'eligible':>9}")
        step = max(1, len(cal.lambdas) // 20)
        for i in range(0, len(cal.lambdas), step):
            elig = cal.ns[i] >= cal.min_routed
            print(f"  {cal.lambdas[i]:7.3f} {cal.ns[i]:9d} {cal.risks[i]:7.3f} "
                  f"{cal.pvalues[i]:9.4f} {str(elig):>9}")
        elig_mask = cal.ns >= cal.min_routed
        if elig_mask.any():
            print(f"\n  min achievable risk among eligible λ: {cal.risks[elig_mask].min():.3f} "
                  f"(alpha={cal.alpha})")
            if cal.risks[elig_mask].min() > cal.alpha:
                print("  >> alpha is NOT achievable with this scorer; raise alpha "
                      "or improve the scorer.")


if __name__ == "__main__":
    main()