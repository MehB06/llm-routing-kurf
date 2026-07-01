"""
Offline preparation for the RouteLLM MF scorer. One script, two subcommands.

  build-inputs     : TRAIN -> (train_fit, train_cal) prompt-disjoint split,
                     then write pairwise_train.json + prompt_embeddings.npy from
                     train_fit ONLY, and train_cal_prompts.json (held-out slice).
  fit-calibrators  : score train_cal with the trained MF model to get raw δ,
                     fit per-model Platt + isotonic (CV-ECE compared), pick the
                     better, save calibrators.pkl (+ optional reliability.png).

Disjointness preserved end to end:
  train_fit -> MF trains | train_cal -> calibrators | CALIB -> LTT | TEST -> eval
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from baselines.ltt_router.splitting import three_way_split
from baselines.ltt_router.routers.embedding_lr import default_embed_fn
from baselines.ltt_router.routers import routellm_mf as R


def _load_records(config):
    from baselines.data_loader import BaselineDataLoader
    return BaselineDataLoader(config).load_all_records()


def _train_split(args):
    """Records -> TRAIN, using the same split the adaptor uses."""
    records = _load_records(args.config)
    train, _calib, _test = three_way_split(
        records, args.train_frac, args.calib_frac, args.seed)
    return records, train


def _prompt_disjoint_split(records, fit_frac, seed):
    """Split records into (fit, cal) at the prompt level, stratified per dataset."""
    import random
    from baselines.ltt_router.splitting import _prompts_by_dataset
    by_ds = defaultdict(list)
    for r in records:
        by_ds[r.dataset_id].append(r)
    fit, cal = [], []
    for _ds, rs in sorted(by_ds.items()):
        p2r, prompts = _prompts_by_dataset(rs)
        idx = list(range(len(prompts)))
        random.Random(seed + 7).shuffle(idx)
        fit_pos = set(idx[:int(len(prompts) * fit_frac)])
        for i, p in enumerate(prompts):
            (fit if i in fit_pos else cal).extend(p2r[p])
    return fit, cal


# --------------------------------------------------------------------------
# build-inputs
# --------------------------------------------------------------------------

def cmd_build_inputs(args):
    _records, train = _train_split(args)
    train_fit, train_cal = _prompt_disjoint_split(train, args.fit_frac, args.seed)

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # embed unique train_fit prompts; row index = idx referenced by pairwise records
    prompts = sorted({r.prompt for r in train_fit})
    p2idx = {p: i for i, p in enumerate(prompts)}
    emb = np.asarray(default_embed_fn(prompts), dtype=np.float32)

    pairwise = _build_pairwise(train_fit, p2idx, args.max_pairs_per_prompt, args.seed)
    if not pairwise:
        raise ValueError("No decisive pairwise records; check TRAIN disagreement.")

    json.dump(pairwise, open(out / "pairwise_train.json", "w"))
    np.save(out / "prompt_embeddings.npy", emb)
    json.dump(sorted({r.prompt for r in train_cal}),
              open(out / "train_cal_prompts.json", "w"))

    print(f"build-inputs done -> {out}")
    print(f"  train_fit: {len(train_fit)} records, {len(prompts)} prompts")
    print(f"  train_cal: {len(train_cal)} records, "
          f"{len({r.prompt for r in train_cal})} prompts (held out)")
    print(f"  pairwise:  {len(pairwise)}   embedding_dim: {emb.shape[1]}")
    print(f"  next: train MF (dim/text_dim={emb.shape[1]}, use_proj=false)")


def _build_pairwise(train_fit, p2idx, cap, seed):
    """Decisive (one-right-one-wrong) model pairs per prompt; ties dropped."""
    cap = None if cap < 0 else cap
    rng = np.random.default_rng(seed)
    by_prompt = defaultdict(list)
    for r in train_fit:
        by_prompt[r.prompt].append((r.model_name, int(r.score == 1.0)))
    pairwise = []
    for p, rows in by_prompt.items():
        pairs = [
            {"model_a": (a if ca > cb else b),
             "model_b": (b if ca > cb else a),
             "winner": "model_a", "idx": p2idx[p]}
            for (a, ca), (b, cb) in combinations(rows, 2)
            if a != b and ca != cb
        ]
        if cap is not None and len(pairs) > cap:
            pairs = [pairs[i] for i in rng.choice(len(pairs), cap, replace=False)]
        pairwise.extend(pairs)
    return pairwise


# --------------------------------------------------------------------------
# fit-calibrators
# --------------------------------------------------------------------------

def cmd_fit_calibrators(args):
    from baselines.adaptors.ltt_adaptor import build_model_specs

    records, train = _train_split(args)
    cal_prompts = set(json.load(open(Path(args.artifacts_dir) / "train_cal_prompts.json")))
    train_cal = [r for r in train if r.prompt in cal_prompts]
    if not train_cal:
        raise ValueError("train_cal empty; rerun build-inputs with same seed/config.")

    models = build_model_specs(records)
    names = [m.name for m in models]
    name_to_idx = {m.name: m.index for m in models}

    # raw δ + per-(prompt, model) correctness over train_cal
    scorer = R.build_routellm_mf_router(models, args.checkpoint, calibrators={})
    by_prompt = defaultdict(list)
    for r in train_cal:
        by_prompt[r.prompt].append(r)
    prompts = list(by_prompt)
    raw = scorer.raw_score_batch(prompts)                 # [Q, N]
    correct = np.zeros_like(raw); evaluated = np.zeros(raw.shape, bool)
    for qi, p in enumerate(prompts):
        for r in by_prompt[p]:
            j = name_to_idx.get(r.model_name)
            if j is not None:
                evaluated[qi, j] = True
                correct[qi, j] = float(r.score == 1.0)

    cals = {names[j]: R.fit_model_calibration(
                raw[evaluated[:, j], j], correct[evaluated[:, j], j],
                names[j], min_samples=args.min_samples)
            for j in range(len(names))}
    method = R.choose_method(cals)
    _print_ece_table(cals, names, method)

    out = Path(args.out).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"method": method,
                 "calibrators": R.calibrator_map(cals, method),
                 "report": {nm: vars(cals[nm]) for nm in names}}, open(out, "wb"))
    print(f"saved -> {out}")

    if args.diagram:
        _reliability(cals, names, raw, correct, evaluated, name_to_idx, args.diagram)
        print(f"reliability diagram -> {args.diagram}")


def _print_ece_table(cals, names, method):
    print(f"{'model':38s} {'n':>6} {'pos':>6} {'ECEraw':>8} {'ECEplatt':>9} {'ECEiso':>8}  deg")
    for nm in names:
        c = cals[nm]
        print(f"{nm:38s} {c.n:6d} {c.pos_rate:6.3f} "
              f"{c.ece_raw:8.4f} {c.ece_platt:9.4f} {c.ece_isotonic:8.4f}  {c.degenerate}")
    nd = [c for c in cals.values() if not c.degenerate]
    if nd:
        m = lambda a: np.nanmean([getattr(c, a) for c in nd])
        print(f"\nmean ECE  raw={m('ece_raw'):.4f}  platt={m('ece_platt'):.4f}  "
              f"iso={m('ece_isotonic'):.4f}")
    print(f"chosen method: {method}")


def _reliability(cals, names, raw, correct, evaluated, name_to_idx, path, n_bins=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nd = [nm for nm in names if not cals[nm].degenerate]
    ncol = min(4, max(1, len(nd))); nrow = (len(nd) + ncol - 1) // ncol
    fig, ax = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
    edges = np.linspace(0, 1, n_bins + 1); ctr = (edges[:-1] + edges[1:]) / 2
    for k, nm in enumerate(nd):
        a = ax[k // ncol][k % ncol]; j = name_to_idx[nm]; ev = evaluated[:, j]
        d, y = raw[ev, j], correct[ev, j]
        for probs, lab in [(1 / (1 + np.exp(-d)), "raw"),
                           (cals[nm].platt(d), "platt"), (cals[nm].isotonic(d), "iso")]:
            accs = [y[(probs >= lo) & (probs < hi)].mean()
                    if ((probs >= lo) & (probs < hi)).any() else np.nan
                    for lo, hi in zip(edges[:-1], edges[1:])]
            a.plot(ctr, accs, marker="o", ms=3, label=lab)
        a.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
        a.set_title(nm, fontsize=8); a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.tick_params(labelsize=6)
        if k == 0:
            a.legend(fontsize=6)
    for k in range(len(nd), nrow * ncol):
        ax[k // ncol][k % ncol].axis("off")
    fig.suptitle("Reliability: predicted prob vs observed accuracy", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _add_split_args(p):
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--calib-frac", type=float, default=0.2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="RouteLLM MF offline pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-inputs"); _add_split_args(b)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--fit-frac", type=float, default=0.6)
    b.add_argument("--max-pairs-per-prompt", type=int, default=50)
    b.set_defaults(func=cmd_build_inputs)

    f = sub.add_parser("fit-calibrators"); _add_split_args(f)
    f.add_argument("--artifacts-dir", required=True)
    f.add_argument("--checkpoint", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--min-samples", type=int, default=50)
    f.add_argument("--diagram", default="")
    f.set_defaults(func=cmd_fit_calibrators)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()