"""Command-line entry point.

  python -m phantom.pipeline generate            (uses data/aml/*.csv if present, else pure synthetic)
  python -m phantom.pipeline evaluate  --data data/synthetic_seed42 --out outputs/eval
  python -m phantom.pipeline sweep     --out outputs/sweep
  python -m phantom.pipeline score     --data data/synthetic_seed42 --out outputs/demo
  python -m phantom.pipeline blind     --train-data data/synthetic_seed42 --blind-seed 777 --out outputs/blind
  python -m phantom.pipeline inspect-aml --trans HI-Small_Trans.csv --patterns HI-Small_Patterns.txt
  python -m phantom.pipeline load-aml    --trans HI-Small_Trans.csv --patterns HI-Small_Patterns.txt --out data/aml_hi_small
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
from .evaluation import run_experiments, controller_sweep, group_ids, metrics, eval_gt
from .explain import assign_tiers, reason_codes, candidate_clusters, cluster_ring_recovery


def _write_run(ds, F, P, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tiers = assign_tiers(P["fusion_full"], F, ds.horizon)
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


def find_aml_files(root="."):
    """Look for IBM AML-Data files in <root>/data/aml (any *trans*.csv and *pattern*.txt; prefers HI-Small)."""
    d = Path(root) / "data" / "aml"
    if not d.exists():
        return None, None

    def pick(files, pref):
        files = sorted(files)
        for f in files:
            if pref in f.name.lower():
                return f
        return files[0] if files else None

    return (pick([f for f in d.glob("*.csv") if "trans" in f.name.lower()], "hi-small"),
            pick([f for f in d.glob("*.txt") if "pattern" in f.name.lower()], "hi-small"))


def _gen_synthetic(out, seed=42, n_companies=4000, obs=0.5):
    ds = generate(GenConfig(seed=seed, n_companies=n_companies, controller_obs_rate=obs))
    ds.save(out)
    print(f"[synthetic] saved {out}: {ds.meta['n_rings']} rings, {ds.meta['n_ring_members']} members, "
          f"prevalence {ds.meta['prevalence']:.2%}, typologies {ds.meta['typology_counts']}")
    return ds


def _gen_aml(trans, patterns, out, benign_seeds=3000, ring_fraction=0.025, seed=42, obs=0.5):
    from .amlworld import build_dataset
    ds = build_dataset(trans, patterns, n_benign_seeds=benign_seeds, ring_fraction=ring_fraction,
                       seed=seed, controller_obs_rate=obs)
    ds.save(out)
    print(f"[hybrid] saved {out}: {ds.meta['n_rings']} ring groups, {ds.meta['n_ring_members']} ring accounts "
          f"({ds.meta['prevalence']:.2%} of {ds.meta['n_seeds']} evaluated), {ds.meta['n_fringe']} fringe accounts; "
          f"types {ds.meta['typology_counts']}")
    return ds


def cmd_generate(a):
    """--source auto (default): use IBM AML-Data from data/aml if present, otherwise pure synthetic."""
    trans, pats = (a.aml_trans, a.aml_patterns) if a.aml_trans else find_aml_files(".")
    source = a.source if a.source != "auto" else ("aml" if trans else "synthetic")
    if source == "aml":
        if not trans:
            raise SystemExit("No IBM files found. Put HI-Small_Trans.csv (and HI-Small_Patterns.txt) in data/aml/ "
                             "or pass --aml-trans / --aml-patterns.")
        print(f"using IBM AML-Data: {trans}" + (f" + {pats}" if pats else " (no patterns file)"))
        _gen_aml(str(trans), str(pats) if pats else None, a.out or "data/aml_hi_small",
                 a.benign_seeds, a.ring_fraction, a.seed, a.controller_obs_rate)
    else:
        print("no IBM files found in data/aml: generating pure synthetic data" if a.source == "auto" else "pure synthetic data")
        _gen_synthetic(a.out or "data/synthetic_seed42", a.seed, a.n_companies, a.controller_obs_rate)


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
    gt = eval_gt(ds)
    F = build_features(ds).loc[gt.entity_id]
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
    gt = eval_gt(train)
    F = build_features(train).loc[gt.entity_id]
    model = PhantomModel(seed=0).fit(F, gt.synthetic_label.to_numpy(), group_ids(gt).to_numpy())
    cfg = GenConfig(**{**train.meta["config"], "seed": a.blind_seed})
    blind = generate(cfg)
    blind_dir = Path(a.out) / "blind_dataset"
    blind.save(blind_dir)
    gtb = eval_gt(blind)
    Fb = build_features(blind).loc[gtb.entity_id]
    P = model.predict(Fb)
    scores, clusters = _write_run(blind, Fb, P, a.out)
    print("BLIND (unseen seed):", {k: round(v, 3) for k, v in
          metrics(gtb.synthetic_label.to_numpy(), P["fusion_full"], gtb.ring_id.to_numpy(dtype=object)).items()
          if k in ("pr_auc", "precision@npos", "recall@2npos", "prevalence")})
    print("candidate clusters:", cluster_ring_recovery(clusters, gtb))
    Path(a.out, "dataset_path.txt").write_text(str(blind_dir.resolve()), encoding="utf-8")


def cmd_inspect_aml(a):
    from .amlworld import inspect_aml
    for k, v in inspect_aml(a.trans, a.patterns).items():
        print(f"{k}: {v}")


def cmd_load_aml(a):
    from .amlworld import build_dataset
    ds = build_dataset(a.trans, a.patterns, n_benign_seeds=a.benign_seeds, ring_fraction=a.ring_fraction,
                       seed=a.seed, controller_obs_rate=a.controller_obs_rate)
    ds.save(a.out)
    print(f"saved {a.out}: {ds.meta['n_rings']} ring groups, {ds.meta['n_ring_members']} ring accounts "
          f"({ds.meta['prevalence']:.2%} of {ds.meta['n_seeds']} evaluated), {ds.meta['n_fringe']} fringe accounts; "
          f"types {ds.meta['typology_counts']}")


def cmd_all(a):
    """Run everything. If IBM files are in data/aml, the hybrid dataset is built and evaluated too."""
    d = Path(a.root)
    rep = 1 if a.quick else 5
    syn = str(d / "data/synthetic_seed42")
    _gen_synthetic(syn, 42, a.n_companies, 0.5)
    trans, pats = find_aml_files(d)
    aml = None
    if trans:
        print(f"\nIBM AML-Data found: {trans}")
        try:
            aml = str(d / "data/aml_hi_small")
            _gen_aml(str(trans), str(pats) if pats else None, aml, a.aml_benign_seeds, a.aml_ring_fraction, 42, 0.5)
        except Exception as e:                      # keep going with synthetic-only
            aml = None
            print(f"\n!! Could not build the hybrid dataset: {e}\n   Run `python -m phantom.pipeline inspect-aml --trans {trans}"
                  f"{' --patterns ' + str(pats) if pats else ''}` and send me the output. Continuing with synthetic data only.\n")
    else:
        print("\n(no IBM files in data/aml - skipping the hybrid dataset. Put HI-Small_Trans.csv and HI-Small_Patterns.txt there to enable it.)")
    runs = [("synthetic", syn, "outputs/eval", "outputs/demo")]
    if aml:
        runs.append(("hybrid (IBM AML-Data + our layers)", aml, "outputs/eval_aml", "outputs/demo_aml"))
    for label, data, ev, demo in runs:
        print(f"\n===== {label}: evaluate =====")
        cmd_evaluate(argparse.Namespace(data=data, out=str(d / ev), repeats=rep))
        print(f"\n===== {label}: score =====")
        cmd_score(argparse.Namespace(data=data, out=str(d / demo)))
    print("\n===== synthetic: controller sweep =====")
    cmd_sweep(argparse.Namespace(out=str(d / "outputs/sweep"), seed=42, n_companies=a.sweep_companies, repeats=1 if a.quick else 3))
    print("\n===== synthetic: blind test on an unseen seed =====")
    cmd_blind(argparse.Namespace(train_data=syn, blind_seed=777, out=str(d / "outputs/blind")))
    print("\nDone. Dashboard: streamlit run app.py  (pick the run in the sidebar)")
    print("Results:", ", ".join(f"{lab.split(' ')[0]} -> {ev}" for lab, _, ev, _ in runs))


def main():
    p = argparse.ArgumentParser(prog="phantom")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="generate data; uses IBM files in data/aml if present")
    g.add_argument("--source", choices=["auto", "synthetic", "aml"], default="auto")
    g.add_argument("--out", default=None); g.add_argument("--seed", type=int, default=42)
    g.add_argument("--n-companies", type=int, default=4000)
    g.add_argument("--controller-obs-rate", type=float, default=0.5)
    g.add_argument("--aml-trans", default=None); g.add_argument("--aml-patterns", default=None)
    g.add_argument("--benign-seeds", type=int, default=3000); g.add_argument("--ring-fraction", type=float, default=0.07)
    g.set_defaults(f=cmd_generate)
    e = sub.add_parser("evaluate"); e.add_argument("--data", default="data/synthetic_seed42")
    e.add_argument("--out", default="outputs/eval"); e.add_argument("--repeats", type=int, default=5); e.set_defaults(f=cmd_evaluate)
    s = sub.add_parser("sweep"); s.add_argument("--out", default="outputs/sweep"); s.add_argument("--seed", type=int, default=42)
    s.add_argument("--n-companies", type=int, default=2500); s.add_argument("--repeats", type=int, default=3); s.set_defaults(f=cmd_sweep)
    c = sub.add_parser("score"); c.add_argument("--data", default="data/synthetic_seed42")
    c.add_argument("--out", default="outputs/demo"); c.set_defaults(f=cmd_score)
    b = sub.add_parser("blind"); b.add_argument("--train-data", default="data/synthetic_seed42")
    b.add_argument("--blind-seed", type=int, default=777); b.add_argument("--out", default="outputs/blind"); b.set_defaults(f=cmd_blind)
    i = sub.add_parser("inspect-aml"); i.add_argument("--trans", required=True)
    i.add_argument("--patterns", default=None); i.set_defaults(f=cmd_inspect_aml)
    l = sub.add_parser("load-aml"); l.add_argument("--trans", required=True); l.add_argument("--patterns", default=None)
    l.add_argument("--out", default="data/aml_hi_small"); l.add_argument("--benign-seeds", type=int, default=3000)
    l.add_argument("--ring-fraction", type=float, default=0.07); l.add_argument("--seed", type=int, default=42)
    l.add_argument("--controller-obs-rate", type=float, default=0.5); l.set_defaults(f=cmd_load_aml)
    al = sub.add_parser("all"); al.add_argument("--root", default=".")
    al.add_argument("--quick", action="store_true", help="fewer repeats, for a fast check")
    al.add_argument("--n-companies", type=int, default=4000); al.add_argument("--sweep-companies", type=int, default=2500)
    al.add_argument("--aml-benign-seeds", type=int, default=3000); al.add_argument("--aml-ring-fraction", type=float, default=0.07)
    al.set_defaults(f=cmd_all)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
