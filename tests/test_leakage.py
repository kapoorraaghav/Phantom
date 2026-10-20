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


def test_company_names_are_unique_deterministic_and_label_free():
    from phantom.names import make_company_names
    a = small()
    b = small()
    assert a.companies.name.is_unique and list(a.companies.name) == list(b.companies.name)
    # a pure function of (seed, sector): ring members are not named differently
    assert list(a.companies.name) == make_company_names(1, a.companies.sector)


def test_network_view_stays_readable_for_hub_companies():
    from phantom.graph import build_graph, network_view
    ds = small()
    G = build_graph(ds, include_accounts=False)
    hub = ds.invoices.groupby("issuer_id").receiver_id.nunique().idxmax()
    H = network_view(G, hub, hops=1, n_partners=5)
    assert H.number_of_nodes() <= 40
    assert not any(d.get("ntype") == "Address" and H.in_degree(n) < 2 for n, d in H.nodes(data=True))


def test_thin_data_rule_scales_with_horizon():
    import pandas as pd
    from phantom.explain import assign_tiers
    F = pd.DataFrame({"n_tx": [5.0], "age_days": [900.0], "ev_n_links": [2.0]}, index=["x"])
    P = pd.Series([0.1], index=["x"])
    assert assign_tiers(P, F, horizon=365)["x"] == "Needs more data"     # 5 events is thin over a year
    assert assign_tiers(P, F, horizon=12)["x"] == "Low"                  # but fine over 12 days


def test_transfers_only_view_removes_all_identity_information():
    F = build_features(small(), use_controller_edges=False, use_contact_links=False)
    assert F[["addr_group_size", "contact_group_size", "idcluster_internal_share", "idcluster_density"]].to_numpy().sum() == 0
    assert (F["idcluster_size"] == 1).all() and (F["ctrl_cluster_size"] == 1).all()


def test_tree_view_structure_and_caps():
    from phantom.graph import build_graph, network_view, tree_view
    ds = small()
    G = build_graph(ds, include_accounts=False)
    c = ds.invoices.groupby("issuer_id").receiver_id.nunique().idxmax()
    H = network_view(G, c, hops=1, n_partners=5)
    nodes, edges = tree_view(H, c, max_children=3)
    ids = {n["id"] for n in nodes}
    assert "root" in ids and all(p in ids and ch in ids for p, ch, _ in edges)
    per_branch = {}
    for n in nodes:
        if n["depth"] == 2:
            per_branch[n["id"].split(":")[0]] = per_branch.get(n["id"].split(":")[0], 0) + (n["kind"] == "company")
    assert all(v <= 3 for v in per_branch.values())
    assert all(abs(n["y"]) < 1e6 for n in nodes)
