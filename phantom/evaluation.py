"""Splits, metrics and the experiment suite (ablations, split diagnostics, robustness)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .config import GenConfig
from .features import build_features
from .generator import generate
from .models import PhantomModel

VARIANTS = ["rules", "if_behaviour", "xgb_behaviour", "xgb_graph", "fusion_behaviour",
            "fusion_graph", "fusion_full", "fusion_full_uncalibrated"]


def eval_gt(ds):
    """Ground truth restricted to entities that are evaluated (all of them for the pure-synthetic
    data; only the sampled 'seed' accounts for AML-Data, whose fringe accounts give graph context)."""
    gt = ds.ground_truth
    if "in_eval" in gt.columns:
        gt = gt[gt.in_eval.astype(bool)]
    return gt.reset_index(drop=True)


def group_ids(gt: pd.DataFrame) -> pd.Series:
    """Leakage groups: ring id for ring members, group id for benign multi-company owners,
    otherwise the company itself."""
    g = gt.set_index("entity_id")
    return g.ring_id.fillna(g.group_id).fillna(pd.Series(g.index, index=g.index))


def ring_grouped_split(gt, test_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    groups = group_ids(gt)
    uniq = groups.unique()
    ring_groups = [x for x in uniq if str(x).startswith("R") and not str(x).startswith("C")]
    ring_groups = [x for x in ring_groups if x in set(gt.ring_id.dropna())]
    other = [x for x in uniq if x not in set(ring_groups)]
    n_r = max(1, int(round(test_frac * len(ring_groups))))
    test_g = set(rng.choice(ring_groups, n_r, replace=False)) | \
        set(rng.choice(other, int(test_frac * len(other)), replace=False))
    test = groups.isin(test_g).to_numpy()
    return np.where(~test)[0], np.where(test)[0]


def metrics(y, score, ring_ids=None):
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    npos = int(y.sum())
    n = len(y)
    prev = npos / n
    order = np.argsort(-score, kind="stable")
    out = {"prevalence": prev, "pr_auc": average_precision_score(y, score),
           "pr_auc_lift": average_precision_score(y, score) / prev if prev else np.nan}
    for label, K in (("npos", npos), ("2npos", 2 * npos)):
        top = order[:K]
        tp = y[top].sum()
        out[f"precision@{label}"] = tp / K
        out[f"recall@{label}"] = tp / npos
        if label == "2npos":
            out["fpr@top2npos"] = (K - tp) / max(n - npos, 1)
        if ring_ids is not None:
            ring_ids = np.asarray(ring_ids, dtype=object)
            rings = [r for r in set(ring_ids) if isinstance(r, str)]
            topset = set(top)
            hit = maj = 0
            for r in rings:
                members = np.where(ring_ids == r)[0]
                inn = sum(1 for m in members if m in topset)
                hit += inn >= 1
                maj += inn >= 0.5 * len(members)
            out[f"ring_any@{label}"] = hit / max(len(rings), 1)
            out[f"ring_majority@{label}"] = maj / max(len(rings), 1)
    return out


def _run(F_tr, y_tr, g_tr, F_te, y_te, ring_te, seed):
    model = PhantomModel(seed=seed).fit(F_tr, y_tr, g_tr)
    P = model.predict(F_te)
    return {v: metrics(y_te, P[v], ring_te) for v in VARIANTS}


def run_experiments(ds, out_dir, n_repeats=5, seed=0, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = eval_gt(ds)
    ids = gt.entity_id.to_numpy()
    y = gt.synthetic_label.to_numpy()
    ring = gt.ring_id.to_numpy(dtype=object)
    groups = group_ids(gt).to_numpy()
    H = ds.horizon
    rows = []

    def add(setting, res, rep):
        for v, m in res.items():
            rows.append(dict(setting=setting, variant=v, repeat=rep, **m))

    log("features: full window"); F = build_features(ds).loc[ids]
    log("features: no controller edges"); F_nc = build_features(ds, use_controller_edges=False).loc[ids]
    log("features: with proposed (fuzzy) links merged"); F_fz = build_features(ds, include_proposed=True).loc[ids]
    log("features: transfers only (no controller, address or phone links)")
    F_io = build_features(ds, use_controller_edges=False, use_contact_links=False).loc[ids]
    log("features: temporal windows")
    F_w1 = build_features(ds, 0, H // 2).loc[ids]; F_w2 = build_features(ds, H // 2, H).loc[ids]

    for rep in range(n_repeats):
        s = seed + rep
        tr, te = ring_grouped_split(gt, 0.3, s)
        log(f"repeat {rep}: grouped split train={len(tr)} test={len(te)} test_pos={y[te].sum()}")
        add("ring_grouped (headline)", _run(F.iloc[tr], y[tr], groups[tr], F.iloc[te], y[te], ring[te], s), rep)
        add("ring_grouped, no controller edges", _run(F_nc.iloc[tr], y[tr], groups[tr], F_nc.iloc[te], y[te], ring[te], s), rep)
        add("ring_grouped, TRANSFERS ONLY (no controller/address/phone links)",
            _run(F_io.iloc[tr], y[tr], groups[tr], F_io.iloc[te], y[te], ring[te], s), rep)
        add("ring_grouped, fuzzy links merged", _run(F_fz.iloc[tr], y[tr], groups[tr], F_fz.iloc[te], y[te], ring[te], s), rep)
        add("temporal+grouped (train early window, test late window)",
            _run(F_w1.iloc[tr], y[tr], groups[tr], F_w2.iloc[te], y[te], ring[te], s), rep)
        # random split diagnostic: ring members leak between train and test
        rng = np.random.default_rng(s)
        mask = rng.random(len(y)) < 0.3
        rtr, rte = np.where(~mask)[0], np.where(mask)[0]
        if y[rte].sum() > 0:
            add("RANDOM split (diagnostic only, leaks rings)",
                _run(F.iloc[rtr], y[rtr], np.arange(len(y))[rtr], F.iloc[rte], y[rte], ring[rte], s), rep)

    # hold out each typology
    typ = gt.typology.to_numpy()
    rng = np.random.default_rng(seed)
    ring_ct = gt[gt.ring_id.notna()].groupby("typology").ring_id.nunique()
    for t in sorted(ring_ct[ring_ct >= 2].index):      # need >= 2 rings of a type to hold it out meaningfully
        te_mask = (typ == t) | ((typ == "none") & (rng.random(len(y)) < 0.3))
        # keep benign multi-company owners together
        te_groups = set(groups[te_mask & (typ != "none")])
        te_mask = te_mask | np.isin(groups, list(te_groups))
        tr, te = np.where(~te_mask)[0], np.where(te_mask)[0]
        add(f"holdout typology = {t}", _run(F.iloc[tr], y[tr], groups[tr], F.iloc[te], y[te], ring[te], seed), 0)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "experiments_raw.csv", index=False)
    summary = summarise(df)
    summary.to_csv(out_dir / "experiments_summary.csv")
    (out_dir / "experiments_summary.md").write_text(to_markdown(summary, ds.meta), encoding="utf-8")
    return df, summary


KEYS = ["pr_auc", "pr_auc_lift", "precision@npos", "recall@2npos", "fpr@top2npos", "ring_any@2npos", "ring_majority@2npos"]


def summarise(df):
    g = df.groupby(["setting", "variant"])[KEYS + ["prevalence"]]
    mean, std = g.mean(), g.std().fillna(0)
    out = mean.copy()
    for k in KEYS:
        out[k] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[k], std[k])]
    out["prevalence"] = mean["prevalence"].round(4)
    return out


def to_markdown(summary, meta):
    lines = ["# PHANTOM experiment summary", "",
             "**All numbers are on SYNTHETIC data and describe the generated scenarios only. "
             "They are not real-world performance.**", "",
             f"Rings: {meta['n_rings']} | ring members: {meta['n_ring_members']} | "
             f"prevalence: {meta['prevalence']:.3%} | typologies: {meta['typology_counts']}", "",
             "A random ranking has PR-AUC ≈ prevalence and Precision@K ≈ prevalence.", ""]
    for setting, sub in summary.groupby(level=0):
        lines += [f"## {setting}", "", sub.droplevel(0).to_markdown(), ""]
    return "\n".join(lines)


def controller_sweep(base_cfg: GenConfig, rates=(0.0, 0.2, 0.5, 0.8, 1.0), n_repeats=3, out_dir="outputs/sweep", log=print):
    """How does detection degrade as controller edges get sparser? (answers 'is the task trivial?')"""
    rows = []
    for r in rates:
        cfg = GenConfig(**{**base_cfg.to_dict(), "controller_obs_rate": r})
        log(f"sweep: controller_obs_rate={r}")
        ds = generate(cfg)
        F = build_features(ds, use_controller_edges=r > 0)
        gt = ds.ground_truth
        y = gt.synthetic_label.to_numpy(); ring = gt.ring_id.to_numpy(dtype=object); groups = group_ids(gt).to_numpy()
        for rep in range(n_repeats):
            tr, te = ring_grouped_split(gt, 0.3, rep)
            res = _run(F.iloc[tr], y[tr], groups[tr], F.iloc[te], y[te], ring[te], rep)
            for v in ("xgb_behaviour", "xgb_graph", "fusion_full"):
                rows.append(dict(controller_obs_rate=r, variant=v, repeat=rep, **res[v]))
    df = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "controller_sweep_raw.csv", index=False)
    s = df.groupby(["controller_obs_rate", "variant"])[["pr_auc", "recall@2npos", "ring_any@2npos"]].mean().round(3)
    s.to_csv(Path(out_dir) / "controller_sweep.csv")
    return s
