"""Guards against label leakage and basic pipeline sanity. Run: pytest -q"""
import inspect
import numpy as np
from phantom import features, models
from phantom.config import GenConfig
from phantom.dataset import LABEL_TABLES
from phantom.features import ALL_COLS, build_features
from phantom.generator import generate
from phantom.evaluation import ring_grouped_split, group_ids


def small():
    return generate(GenConfig(n_companies=800, seed=1))


def test_feature_code_never_touches_label_tables():
    src = inspect.getsource(features) + inspect.getsource(models)
    for t in LABEL_TABLES:
        assert f"ds.{t}" not in src, f"feature/model code reads {t}"


def test_no_label_columns_in_features():
    F = build_features(small())
    banned = {"ring_id", "controller_id", "typology", "injection_seed", "synthetic_label", "archetype", "group_id"}
    assert not banned & set(F.columns)
    assert list(F.columns) == ALL_COLS


def test_features_deterministic():
    a, b = build_features(small()), build_features(small())
    assert a.equals(b)


def test_grouped_split_keeps_rings_whole():
    ds = small()
    gt = ds.ground_truth
    tr, te = ring_grouped_split(gt, 0.3, 0)
    tr_rings = set(gt.ring_id.iloc[tr].dropna())
    te_rings = set(gt.ring_id.iloc[te].dropna())
    assert te_rings and not (tr_rings & te_rings)


def test_controller_edges_are_partial_and_noisy():
    ds = small()
    true = set(zip(ds.true_controller_edges.person_id, ds.true_controller_edges.company_id))
    obs = set(zip(ds.relationships.source_id, ds.relationships.target_id))
    assert 0.2 < len(obs & true) / len(true) < 0.8      # partial
    assert len(obs - true) > 0                          # noisy (wrong links present)


def test_no_controller_ablation_zeroes_controller_features():
    F = build_features(small(), use_controller_edges=False)
    # with no controller edges every company is its own cluster (size 1); everything else is 0
    assert (F["ctrl_cluster_size"] == 1).all()
    assert F[[c for c in features.CONTROLLER_COLS if c != "ctrl_cluster_size"]].to_numpy().sum() == 0


def test_benign_lookalikes_exist():
    gt = small().ground_truth
    assert {"seasonal", "cash_heavy", "wholesaler"} <= set(gt[gt.synthetic_label == 0].archetype)
    assert gt[(gt.synthetic_label == 0) & gt.group_id.notna()].shape[0] > 0   # coworking / business groups
