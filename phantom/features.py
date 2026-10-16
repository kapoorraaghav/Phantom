"""Feature engineering. Reads ONLY observable tables (never ground_truth / true_controller_edges).

`build_features(ds, t0, t1, use_controller_edges, include_proposed)` returns one row per company,
using only events inside [t0, t1) and relationship records that started before t1.
"""
from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
from .config import ITEM_SECTOR, CONTROLLER_TYPES
from .dataset import Dataset
from .entity_resolution import resolve_identifiers

BEHAVIOUR_COLS = [
    "n_inv_out", "n_inv_in", "log_amt_out", "log_amt_in", "round_share_out", "mismatch_share_out",
    "lag_mean", "lag_std", "flow_balance", "log_in_out_ratio", "median_dwell_days", "short_dwell_share",
    "n_cpty", "cpty_hhi", "week_cv", "max_week_share", "dormancy_frac", "longest_dormant_frac",
    "age_days", "active_span_days", "vol_vs_declared", "n_tx",
]
GRAPH_TX_COLS = ["in_deg", "out_deg", "reciprocal_partners", "pagerank", "clustering",
                 "cycles3", "cycles4", "tight_cycles"]
IDENTITY_COLS = ["addr_group_size", "contact_group_size", "idcluster_size", "idcluster_internal_share",
                 "idcluster_density"]
CONTROLLER_COLS = ["ctrl_n_controllers", "ctrl_max_reach", "ctrl_cluster_size", "ctrl_cluster_internal_share"]
GRAPH_COLS = GRAPH_TX_COLS + IDENTITY_COLS + CONTROLLER_COLS
EVIDENCE_COLS = ["ev_n_links", "ev_mean_conf", "ev_n_sources", "ev_recency_days", "ev_conflicts"]
ALL_COLS = BEHAVIOUR_COLS + GRAPH_COLS + EVIDENCE_COLS


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def count_cycles(G: nx.DiGraph, maxlen: int = 4, tight_ratio: float = 1.25, min_edge_n: int = 2):
    """Count directed simple cycles of length 3..maxlen through each node.

    A cycle is 'tight' when the median edge amounts differ by at most `tight_ratio` (typical of
    round-tripping). Each cycle is found once, from its smallest node.
    """
    nodes = sorted(G.nodes)
    order = {n: i for i, n in enumerate(nodes)}
    succ = {n: [(m, G[n][m]["med"]) for m in G.successors(n) if m != n and G[n][m]["n"] >= min_edge_n]
            for n in nodes}
    c3, c4, ct = Counter(), Counter(), Counter()

    def dfs(s, cur, path, amts):
        for m, a in succ[cur]:
            if m == s and len(path) >= 3:
                tight = max(amts + [a]) / max(min(amts + [a]), 1e-9) <= tight_ratio
                for v in path:
                    (c3 if len(path) == 3 else c4)[v] += 1
                    if tight:
                        ct[v] += 1
            elif order[m] > order[s] and m not in path and len(path) < maxlen:
                dfs(s, m, path + [m], amts + [a])

    for s in nodes:
        dfs(s, s, [s], [])
    return c3, c4, ct


def _longest_zero_run(row):
    best = cur = 0
    for v in row:
        cur = cur + 1 if v == 0 else 0
        best = max(best, cur)
    return best


def build_features(ds: Dataset, t0: int = 0, t1: int = None, use_controller_edges: bool = True,
                   include_proposed: bool = False) -> pd.DataFrame:
    t1 = ds.horizon if t1 is None else t1
    comp = ds.companies.set_index("company_id")
    ids = comp.index
    F = pd.DataFrame(index=ids)
    inv_all = ds.invoices
    inv = inv_all[(inv_all.day >= t0) & (inv_all.day < t1)].copy()
    tx = ds.transactions[(ds.transactions.day >= t0) & (ds.transactions.day < t1)].copy()
    acc_map = ds.accounts.set_index("account_id")["linked_entity_id"]
    tx["src"] = tx.sender_account.map(acc_map)
    tx["dst"] = tx.receiver_account.map(acc_map)

    # ------------------------------------------------------------------ behaviour
    inv["is_round"] = ((inv.amount % 1000) == 0) & (inv.amount >= 5000)
    inv["mismatch"] = inv.item_category.map(ITEM_SECTOR).values != inv.issuer_id.map(comp.sector).values
    o = inv.groupby("issuer_id").agg(n_inv_out=("invoice_id", "size"), amt_out=("amount", "sum"),
                                     round_share_out=("is_round", "mean"),
                                     mismatch_share_out=("mismatch", "mean"))
    i = inv.groupby("receiver_id").agg(n_inv_in=("invoice_id", "size"), amt_in=("amount", "sum"))
    F = F.join(o).join(i)
    for c in ["n_inv_out", "n_inv_in", "amt_out", "amt_in", "round_share_out", "mismatch_share_out"]:
        F[c] = F[c].fillna(0)
    F["log_amt_out"] = np.log1p(F.amt_out)
    F["log_amt_in"] = np.log1p(F.amt_in)

    inv_day = inv_all.set_index("invoice_id")["day"]
    inv_issuer = inv_all.set_index("invoice_id")["issuer_id"]
    paid = tx.dropna(subset=["invoice_id"]).copy()
    paid["lag"] = paid.t - paid.invoice_id.map(inv_day)
    paid["issuer"] = paid.invoice_id.map(inv_issuer)
    lag = paid.groupby("issuer")["lag"].agg(lag_mean="mean", lag_std="std")
    F = F.join(lag)
    F["lag_mean"] = F.lag_mean.fillna(30.0)
    F["lag_std"] = F.lag_std.fillna(10.0)

    inflow = tx.groupby("dst")["amount"].sum()
    outflow = tx.groupby("src")["amount"].sum()
    F["inflow"] = inflow.reindex(ids).fillna(0)
    F["outflow"] = outflow.reindex(ids).fillna(0)
    mx = np.maximum(F.inflow, F.outflow)
    F["flow_balance"] = np.where(mx > 0, np.minimum(F.inflow, F.outflow) / np.maximum(mx, 1), 0.0)
    F["log_in_out_ratio"] = np.log((F.inflow + 1) / (F.outflow + 1))

    outs = tx[["src", "t"]].rename(columns={"src": "c"}).dropna().sort_values("t")
    ins = tx[["dst", "t"]].rename(columns={"dst": "c"}).dropna().sort_values("t")
    ins["t_in"] = ins["t"]
    mg = pd.merge_asof(outs, ins, on="t", by="c", direction="backward")
    mg["dwell"] = mg.t - mg.t_in
    dw = mg.groupby("c")["dwell"].median()
    F["median_dwell_days"] = dw.reindex(ids).fillna(float(t1 - t0))
    F["short_dwell_share"] = (mg.assign(s=(mg.dwell <= 3).astype(float)).groupby("c")["s"].mean()
                              .reindex(ids).fillna(0))

    a = inv[["issuer_id", "receiver_id", "amount"]].rename(columns={"issuer_id": "c", "receiver_id": "p"})
    b = inv[["receiver_id", "issuer_id", "amount"]].rename(columns={"receiver_id": "c", "issuer_id": "p"})
    pairs_dir = pd.concat([a, b]).groupby(["c", "p"])["amount"].sum().reset_index()
    F["n_cpty"] = pairs_dir.groupby("c")["p"].nunique().reindex(ids).fillna(0)
    tot = pairs_dir.groupby("c")["amount"].transform("sum")
    pairs_dir["share2"] = (pairs_dir.amount / tot) ** 2
    F["cpty_hhi"] = pairs_dir.groupby("c")["share2"].sum().reindex(ids).fillna(1.0)

    ev = pd.concat([tx[["src", "t"]].rename(columns={"src": "c"}), tx[["dst", "t"]].rename(columns={"dst": "c"})]).dropna()
    nweeks = max(1, int(np.ceil((t1 - t0) / 7)))
    ev["w"] = ((ev.t - t0) // 7).astype(int).clip(0, nweeks - 1)
    M = ev.groupby(["c", "w"]).size().unstack(fill_value=0).reindex(index=ids, columns=range(nweeks), fill_value=0)
    Mv = M.to_numpy(dtype=float)
    tot_ev = Mv.sum(1)
    F["n_tx"] = tot_ev
    F["week_cv"] = np.where(Mv.mean(1) > 0, Mv.std(1) / np.maximum(Mv.mean(1), 1e-9), 0)
    F["max_week_share"] = np.where(tot_ev > 0, Mv.max(1) / np.maximum(tot_ev, 1), 0)
    F["dormancy_frac"] = (Mv == 0).mean(1)
    F["longest_dormant_frac"] = np.array([_longest_zero_run(r) for r in Mv]) / nweeks
    first = ev.groupby("c")["t"].min().reindex(ids)
    last = ev.groupby("c")["t"].max().reindex(ids)
    F["active_span_days"] = (last - first).fillna(0)
    F["age_days"] = t1 - comp.reg_day
    vol = np.log1p(F.amt_out + F.amt_in)
    F["vol_vs_declared"] = vol - vol.groupby(comp.declared_size).transform("median")

    # ------------------------------------------------------------------ transaction-graph features
    agg = inv.groupby(["issuer_id", "receiver_id"])["amount"].agg(n="size", med="median", total="sum").reset_index()
    G = nx.DiGraph()
    G.add_nodes_from(ids)
    for r in agg.itertuples(index=False):
        G.add_edge(r.issuer_id, r.receiver_id, n=r.n, med=r.med, total=r.total)
    F["in_deg"] = pd.Series(dict(G.in_degree())).reindex(ids)
    F["out_deg"] = pd.Series(dict(G.out_degree())).reindex(ids)
    F["reciprocal_partners"] = pd.Series({u: sum(1 for v in G.successors(u) if G.has_edge(v, u)) for u in G.nodes}).reindex(ids)
    F["pagerank"] = pd.Series(nx.pagerank(G, weight="total", max_iter=100)).reindex(ids) * len(ids)
    F["clustering"] = pd.Series(nx.clustering(G.to_undirected())).reindex(ids)
    c3, c4, ct = count_cycles(G)
    F["cycles3"] = pd.Series(c3).reindex(ids).fillna(0)
    F["cycles4"] = pd.Series(c4).reindex(ids).fillna(0)
    F["tight_cycles"] = pd.Series(ct).reindex(ids).fillna(0)

    # ------------------------------------------------------------------ identity + controller features
    contacts, _ = resolve_identifiers(ds.company_contacts, include_proposed=include_proposed)
    contacts = contacts[contacts.start_day < t1].set_index("company_id").reindex(ids)
    addr_sz = contacts.groupby("addr_key")["addr_key"].transform("size") - 1
    ph_sz = contacts.groupby("phone_key")["phone_key"].transform("size") - 1
    F["addr_group_size"] = addr_sz.values
    F["contact_group_size"] = ph_sz.values

    rel = ds.relationships
    rel = rel[(rel.relationship_type.isin(CONTROLLER_TYPES)) & (rel.start_day < t1)]
    if not use_controller_edges:
        rel = rel.iloc[0:0]
    uf_all, uf_ctrl = _UF(), _UF()
    for c in ids:
        uf_all.find(c)
        uf_ctrl.find(c)
    for p, c in zip(rel.source_id, rel.target_id):
        uf_all.union(c, p)
        uf_ctrl.union(c, p)
    for col in ("addr_key", "phone_key"):
        for key, grp in contacts.reset_index().groupby(col)["company_id"]:
            g = list(grp)
            for x in g[1:]:
                uf_all.union(g[0], x)
    cl_all = pd.Series({c: uf_all.find(c) for c in ids})
    cl_ctrl = pd.Series({c: uf_ctrl.find(c) for c in ids})
    F["idcluster_size"] = cl_all.map(cl_all.value_counts())
    F["ctrl_cluster_size"] = cl_ctrl.map(cl_ctrl.value_counts())
    per_person = rel.groupby("source_id")["target_id"].nunique()
    rel2 = rel.assign(reach=rel.source_id.map(per_person))
    F["ctrl_max_reach"] = rel2.groupby("target_id")["reach"].max().reindex(ids).fillna(0)
    F["ctrl_n_controllers"] = rel.groupby("target_id")["source_id"].nunique().reindex(ids).fillna(0)
    # flows inside identity clusters
    pp = pairs_dir.copy()
    pp["same_all"] = pp.c.map(cl_all).values == pp.p.map(cl_all).values
    pp["same_ctrl"] = (pp.c.map(cl_ctrl).values == pp.p.map(cl_ctrl).values) & (pp.c.map(cl_ctrl).map(cl_ctrl.value_counts()).values > 1)
    tot2 = pp.groupby("c")["amount"].sum()
    F["idcluster_internal_share"] = (pp[pp.same_all].groupby("c")["amount"].sum() / tot2).reindex(ids).fillna(0)
    F["ctrl_cluster_internal_share"] = (pp[pp.same_ctrl].groupby("c")["amount"].sum() / tot2).reindex(ids).fillna(0)
    # cluster density: distinct directed internal pairs / possible
    di = agg.copy()
    di = di[di.issuer_id.map(cl_all).values == di.receiver_id.map(cl_all).values]
    internal = di.groupby(di.issuer_id.map(cl_all)).size()
    k = cl_all.map(cl_all.value_counts())
    dens = cl_all.map(internal).fillna(0) / np.maximum(k * (k - 1), 1)
    F["idcluster_density"] = np.where(k > 1, dens, 0.0)

    # ------------------------------------------------------------------ evidence features
    sh_a = (F.addr_group_size > 0).astype(int)
    sh_c = (F.contact_group_size > 0).astype(int)
    n_pe = rel.groupby("target_id").size().reindex(ids).fillna(0)
    F["ev_n_links"] = n_pe + sh_a + sh_c
    conf_sum = rel.groupby("target_id")["confidence"].sum().reindex(ids).fillna(0) + sh_a + sh_c
    F["ev_mean_conf"] = np.where(F.ev_n_links > 0, conf_sum / np.maximum(F.ev_n_links, 1), 0.0)
    src_sets = rel.groupby("target_id")["source"].agg(lambda s: set(s)).reindex(ids)
    F["ev_n_sources"] = [len(s) if isinstance(s, set) else 0 for s in src_sets] + sh_a.values + sh_c.values
    F["ev_recency_days"] = (t1 - rel.groupby("target_id")["start_day"].max().reindex(ids)).fillna(3650)
    own_n = rel[rel.relationship_type == "OWNS"].groupby("target_id")["source_id"].nunique().reindex(ids).fillna(0)
    F["ev_conflicts"] = np.maximum(own_n - 1, 0)

    F = F[ALL_COLS].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
    return F
