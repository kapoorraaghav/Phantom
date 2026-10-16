"""Command-line entry point.

  python -m phantom.pipeline generate  --out data/synthetic_seed42
  python -m phantom.pipeline evaluate  --data data/synthetic_seed42 --out outputs/eval
  python -m phantom.pipeline sweep     --out outputs/sweep
  python -m phantom.pipeline score     --data data/synthetic_seed42 --out outputs/demo
  python -m phantom.pipeline blind     --train-data data/synthetic_seed42 --blind-seed 777 --out outputs/blind
  python -m phantom.pipeline all
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from .config import GenConfig
from .dataset import Dataset
from .generator import generate
from .features import build_features
from .models import PhantomModel
from .evaluation import run_experiments, controller_sweep, group_ids, metrics
from .explain import assign_tiers, reason_codes, candidate_clusters, cluster_ring_recovery


def _write_run(ds, F, P, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tiers = assign_tiers(P["fusion_full"], F)
    scores = P.copy()
    scores["tier"] = tiers
    scores["rank"] = scores["fusion_full"].rank(ascending=False, method="first").astype(int)
    scores.index.name = "company_id"
    scores.reset_index().to_parquet(out / "scores.parquet", index=False)
    F.reset_index().rename(columns={"index": "company_id"}).to_parquet(out / "features.parquet", index=False)
    flagged = tiers[tiers.isin(["High", "Review", "Needs more data"])].sort_values().index
    top = scores[scores.tier.isin(["High", "Review"])].sort_values("fusion_full", ascending=False).head(150).index
    reasons = {c: reason_codes(ds, F, c) for c in top}
    (out / "reasons.json").write_text(json.dumps(reasons, indent=1, default=float), encoding="utf-8")
    clusters = candidate_clusters(ds, P["fusion_full"], tiers, F)
    (out / "clusters.json").write_text(json.dumps(clusters, indent=1, default=float), encoding="utf-8")
    return scores, clusters


def cmd_generate(a):
    cfg = GenConfig(seed=a.seed, n_companies=a.n_companies, controller_obs_rate=a.controller_obs_rate)
    ds = generate(cfg)
    ds.save(a.out)
    print(f"saved {a.out}: {ds.meta['n_rings']} rings, {ds.meta['n_ring_members']} members, "
          f"prevalence {ds.meta['prevalence']:.2%}, typologies {ds.meta['typology_counts']}")


def cmd_evaluate(a):
    ds = Dataset.load(a.data)
    _, summary = run_experiments(ds, a.out, n_repeats=a.repeats)
    print(f"\nwritten to {a.out}/experiments_summary.md")


def cmd_sweep(a):
    s = controller_sweep(GenConfig(seed=a.seed, n_companies=a.n_companies), n_repeats=a.repeats, out_dir=a.out)
    print(s)


def cmd_score(a):
    """Out-of-fold scores for every company: nobody is scored by a model that saw their ring."""
    ds = Dataset.load(a.data)
    F = build_features(ds)
    gt = ds.ground_truth
    y, groups = gt.synthetic_label.to_numpy(), group_ids(gt).to_numpy()
    P = None
    for tr, te in GroupKFold(n_splits=5).split(F, y, groups):
        m = PhantomModel(seed=0).fit(F.iloc[tr], y[tr], groups[tr])
        part = m.predict(F.iloc[te])
        P = part if P is None else pd.concat([P, part])
    P = P.loc[F.index]
    scores, clusters = _write_run(ds, F, P, a.out)
    print("overall (out-of-fold, ring-grouped):", {k: round(v, 3) for k, v in
          metrics(y, P["fusion_full"], gt.ring_id.to_numpy(dtype=object)).items() if k in ("pr_auc", "precision@npos", "recall@2npos")})
    print("candidate clusters:", cluster_ring_recovery(clusters, gt))
    print(scores.tier.value_counts().to_dict())
    Path(a.out, "dataset_path.txt").write_text(str(Path(a.data).resolve()), encoding="utf-8")


def cmd_blind(a):
    """Train on one synthetic world, score a *different* world generated from a held-back seed."""
    train = Dataset.load(a.train_data)
    gt = train.ground_truth
    F = build_features(train)
    model = PhantomModel(seed=0).fit(F, gt.synthetic_label.to_numpy(), group_ids(gt).to_numpy())
    cfg = GenConfig(**{**train.meta["config"], "seed": a.blind_seed})
    blind = generate(cfg)
    blind_dir = Path(a.out) / "blind_dataset"
    blind.save(blind_dir)
    Fb = build_features(blind)
    P = model.predict(Fb)
    scores, clusters = _write_run(blind, Fb, P, a.out)
    gtb = blind.ground_truth
    print("BLIND (unseen seed):", {k: round(v, 3) for k, v in
          metrics(gtb.synthetic_label.to_numpy(), P["fusion_full"], gtb.ring_id.to_numpy(dtype=object)).items()
          if k in ("pr_auc", "precision@npos", "recall@2npos", "prevalence")})
    print("candidate clusters:", cluster_ring_recovery(clusters, gtb))
    Path(a.out, "dataset_path.txt").write_text(str(blind_dir.resolve()), encoding="utf-8")


def cmd_all(a):
    d = Path(a.root)
    cmd_generate(argparse.Namespace(out=str(d / "data/synthetic_seed42"), seed=42, n_companies=4000, controller_obs_rate=0.5))
    cmd_evaluate(argparse.Namespace(data=str(d / "data/synthetic_seed42"), out=str(d / "outputs/eval"), repeats=5))
    cmd_sweep(argparse.Namespace(out=str(d / "outputs/sweep"), seed=42, n_companies=2500, repeats=3))
    cmd_score(argparse.Namespace(data=str(d / "data/synthetic_seed42"), out=str(d / "outputs/demo")))
    cmd_blind(argparse.Namespace(train_data=str(d / "data/synthetic_seed42"), blind_seed=777, out=str(d / "outputs/blind")))


def main():
    p = argparse.ArgumentParser(prog="phantom")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--out", default="data/synthetic_seed42")
    g.add_argument("--seed", type=int, default=42); g.add_argument("--n-companies", type=int, default=4000)
    g.add_argument("--controller-obs-rate", type=float, default=0.5); g.set_defaults(f=cmd_generate)
    e = sub.add_parser("evaluate"); e.add_argument("--data", default="data/synthetic_seed42")
    e.add_argument("--out", default="outputs/eval"); e.add_argument("--repeats", type=int, default=5); e.set_defaults(f=cmd_evaluate)
    s = sub.add_parser("sweep"); s.add_argument("--out", default="outputs/sweep"); s.add_argument("--seed", type=int, default=42)
    s.add_argument("--n-companies", type=int, default=2500); s.add_argument("--repeats", type=int, default=3); s.set_defaults(f=cmd_sweep)
    c = sub.add_parser("score"); c.add_argument("--data", default="data/synthetic_seed42")
    c.add_argument("--out", default="outputs/demo"); c.set_defaults(f=cmd_score)
    b = sub.add_parser("blind"); b.add_argument("--train-data", default="data/synthetic_seed42")
    b.add_argument("--blind-seed", type=int, default=777); b.add_argument("--out", default="outputs/blind"); b.set_defaults(f=cmd_blind)
    al = sub.add_parser("all"); al.add_argument("--root", default="."); al.set_defaults(f=cmd_all)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
