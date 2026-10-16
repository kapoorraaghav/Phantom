"""Reason codes, supporting evidence, tiers and candidate clusters.

Every reason code is a template filled with real feature values, and carries the concrete
records (invoices, transactions, edges) that support it.
"""
import networkx as nx
import numpy as np
import pandas as pd
from .config import HIGH_P, REVIEW_P, CONTROLLER_TYPES, ITEM_SECTOR
from .entity_resolution import resolve_identifiers


def assign_tiers(P: pd.Series, F: pd.DataFrame) -> pd.Series:
    """Low / Review / High / Needs more data.

    'Needs more data' when the score cannot be trusted:
      * fewer than 10 transaction events, or less than 60 days of history, OR
      * the score is in the Review..High band but there is no relationship or shared-identifier
        evidence at all (behaviour-only suspicion with nothing to corroborate it).
    """
    tier = np.where(P >= HIGH_P, "High", np.where(P >= REVIEW_P, "Review", "Low")).astype(object)
    thin = (F.n_tx < 10) | (F.age_days < 60)
    no_ev = (F.ev_n_links == 0) & (P >= REVIEW_P) & (P < HIGH_P)
    tier[(thin | no_ev).to_numpy()] = "Needs more data"
    return pd.Series(tier, index=P.index)


def _pct(F, col, v):
    return float((F[col] <= v).mean() * 100)


def tight_cycle_paths(ds, company, t0=0, t1=None, maxlen=4, ratio=1.25, min_n=2, limit=3):
    t1 = ds.horizon if t1 is None else t1
    inv = ds.invoices[(ds.invoices.day >= t0) & (ds.invoices.day < t1)]
    agg = inv.groupby(["issuer_id", "receiver_id"])["amount"].agg(n="size", med="median").reset_index()
    agg = agg[agg.n >= min_n]
    G = nx.DiGraph()
    for r in agg.itertuples():
        G.add_edge(r.issuer_id, r.receiver_id, med=r.med)
    out = []

    def dfs(cur, path, amts):
        if len(out) >= limit:
            return
        for m in G.successors(cur) if cur in G else []:
            a = G[cur][m]["med"]
            if m == company and len(path) >= 3:
                al = amts + [a]
                if max(al) / min(al) <= ratio:
                    out.append(path + [company])
            elif m not in path and len(path) < maxlen:
                dfs(m, path + [m], amts + [a])

    if company in G:
        dfs(company, [company], [])
    return out


def reason_codes(ds, F: pd.DataFrame, company: str, max_codes=6):
    """Return [{code, text, features, support}] for one company."""
    r = F.loc[company]
    comp = ds.companies.set_index("company_id")
    codes = []
    cyc = tight_cycle_paths(ds, company) if r.tight_cycles > 0 else []
    if r.tight_cycles > 0:
        codes.append(dict(code="TIGHT_CYCLE",
                          text=f"Sits on {int(r.tight_cycles)} circular invoice flow(s) of length 3-4 with near-identical amounts.",
                          features={"tight_cycles": r.tight_cycles},
                          support=[" → ".join(p) for p in cyc]))
    if r.round_share_out > 0.3 and r.n_inv_out >= 5:
        inv = ds.invoices[(ds.invoices.issuer_id == company) & (ds.invoices.amount % 1000 == 0)].head(3)
        codes.append(dict(code="ROUND_VALUES",
                          text=f"{r.round_share_out:.0%} of issued invoices are round-thousand amounts (population {_pct(F, 'round_share_out', r.round_share_out):.0f}th percentile).",
                          features={"round_share_out": r.round_share_out},
                          support=[f"{x.invoice_id}: {x.amount:,.0f} to {x.receiver_id}" for x in inv.itertuples()]))
    if r.median_dwell_days < 3 and r.flow_balance > 0.8 and r.n_inv_in >= 5:
        acc = ds.accounts[ds.accounts.linked_entity_id == company].account_id
        tx = ds.transactions
        i_ = tx[tx.receiver_account.isin(acc)].nlargest(2, "amount")
        o_ = tx[tx.sender_account.isin(acc)].nlargest(2, "amount")
        codes.append(dict(code="PASS_THROUGH",
                          text=f"Outflow is {r.flow_balance:.0%} of inflow and funds leave a median {r.median_dwell_days:.1f} days after arriving.",
                          features={"flow_balance": r.flow_balance, "median_dwell_days": r.median_dwell_days},
                          support=[f"in {x.transaction_id}: {x.amount:,.0f}" for x in i_.itertuples()] +
                                  [f"out {x.transaction_id}: {x.amount:,.0f}" for x in o_.itertuples()]))
    if r.max_week_share > 0.30 and r.dormancy_frac > 0.5:
        codes.append(dict(code="DORMANT_THEN_SPIKE",
                          text=f"{r.dormancy_frac:.0%} of weeks are inactive, yet {r.max_week_share:.0%} of all activity falls in one week.",
                          features={"dormancy_frac": r.dormancy_frac, "max_week_share": r.max_week_share}, support=[]))
    if r.mismatch_share_out > 0.3:
        inv = ds.invoices[ds.invoices.issuer_id == company]
        bad = inv[inv.item_category.map(ITEM_SECTOR) != comp.loc[company, "sector"]].head(3)
        codes.append(dict(code="SECTOR_MISMATCH",
                          text=f"{r.mismatch_share_out:.0%} of issued invoices are for items outside the declared sector ({comp.loc[company, 'sector']}).",
                          features={"mismatch_share_out": r.mismatch_share_out},
                          support=[f"{x.invoice_id}: {x.item_category}" for x in bad.itertuples()]))
    if r.ctrl_max_reach >= 3:
        rel = ds.relationships
        mine = rel[(rel.target_id == company) & rel.relationship_type.isin(CONTROLLER_TYPES)]
        sup = []
        for p in mine.itertuples():
            others = rel[(rel.source_id == p.source_id) & (rel.target_id != company)]
            sup.append(f"{p.source_id} ({p.relationship_type}, {p.source}, conf {p.confidence:.2f}) also linked to: "
                       + ", ".join(others.target_id.head(6)))
        codes.append(dict(code="SHARED_CONTROLLER",
                          text=f"An observed controller is linked to {int(r.ctrl_max_reach)} companies.",
                          features={"ctrl_max_reach": r.ctrl_max_reach}, support=sup[:3]))
    if r.idcluster_size >= 2 and r.idcluster_size <= 12 and r.idcluster_internal_share > 0.2:
        codes.append(dict(code="INSULAR_CLUSTER",
                          text=f"Linked by shared identifiers to {int(r.idcluster_size) - 1} other companies and {r.idcluster_internal_share:.0%} of its invoice value stays inside that cluster.",
                          features={"idcluster_size": r.idcluster_size, "idcluster_internal_share": r.idcluster_internal_share},
                          support=[]))
    if r.contact_group_size > 0 or (0 < r.addr_group_size <= 10):
        contacts, _ = resolve_identifiers(ds.company_contacts)
        mine = contacts[contacts.company_id == company].iloc[0]
        same_a = contacts[(contacts.addr_key == mine.addr_key) & (contacts.company_id != company)].company_id
        same_p = contacts[(contacts.phone_key == mine.phone_key) & (contacts.company_id != company)].company_id
        sup = []
        if len(same_a):
            sup.append("same address as: " + ", ".join(same_a.head(8)))
        if len(same_p):
            sup.append("same phone as: " + ", ".join(same_p.head(8)))
        if sup:
            codes.append(dict(code="SHARED_IDENTIFIER",
                              text="Shares a registered address and/or phone number with other companies.",
                              features={"addr_group_size": r.addr_group_size, "contact_group_size": r.contact_group_size},
                              support=sup))
    return codes[:max_codes]


def candidate_clusters(ds, scores: pd.Series, tiers: pd.Series, F: pd.DataFrame, min_size=2):
    """Group flagged companies (High/Review) into candidate rings.

    Two flagged companies are connected if they share an observed controller, address or phone,
    or have a direct invoice relationship. Returns a list of clusters with their supporting edges.
    """
    flagged = set(tiers[tiers.isin(["High", "Review"])].index)
    G = nx.Graph()
    G.add_nodes_from(flagged)
    edges = {}

    def link(a, b, why):
        if a in flagged and b in flagged and a != b:
            G.add_edge(a, b)
            edges.setdefault(frozenset((a, b)), set()).add(why)

    rel = ds.relationships[ds.relationships.relationship_type.isin(CONTROLLER_TYPES)]
    for p, grp in rel.groupby("source_id"):
        ts = list(grp.target_id)
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                link(ts[i], ts[j], f"shared controller {p}")
    contacts, _ = resolve_identifiers(ds.company_contacts)
    contacts = contacts[contacts.company_id.isin(flagged)]
    for col, label in (("addr_key", "shared address"), ("phone_key", "shared phone")):
        for _, grp in contacts.groupby(col):
            ts = list(grp.company_id)
            for i in range(len(ts)):
                for j in range(i + 1, len(ts)):
                    link(ts[i], ts[j], label)
    inv = ds.invoices[ds.invoices.issuer_id.isin(flagged) & ds.invoices.receiver_id.isin(flagged)]
    for (a, b), grp in inv.groupby(["issuer_id", "receiver_id"]):
        if len(grp) >= 2:
            link(a, b, f"{len(grp)} invoices {a}→{b}")
    out = []
    for k, comp in enumerate(sorted(nx.connected_components(G), key=len, reverse=True)):
        if len(comp) < min_size:
            continue
        sub = G.subgraph(comp)
        out.append(dict(cluster_id=f"K{k:03d}", members=sorted(comp), size=len(comp),
                        mean_score=float(scores[list(comp)].mean()),
                        evidence=[dict(a=min(e), b=max(e), why=sorted(edges[frozenset(e)])) for e in sub.edges()]))
    return out


def cluster_ring_recovery(clusters, gt: pd.DataFrame):
    """Ring-level recall: a ring is 'identified' if some cluster contains >= 50% of its members."""
    rings = gt[gt.ring_id.notna()].groupby("ring_id")["entity_id"].apply(set)
    ok = 0
    for rid, members in rings.items():
        if any(len(members & set(c["members"])) >= 0.5 * len(members) for c in clusters):
            ok += 1
    pure = [np.mean([m in set(gt[gt.synthetic_label == 1].entity_id) for m in c["members"]]) for c in clusters]
    return dict(rings=len(rings), rings_identified=ok, ring_recall=ok / max(len(rings), 1),
                n_clusters=len(clusters), mean_cluster_purity=float(np.mean(pure)) if pure else 0.0)
