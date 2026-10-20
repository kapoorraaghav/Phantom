"""Heterogeneous, typed, provenance-carrying graph (for exploration, traversal and the dashboard)."""
import networkx as nx
import pandas as pd
from .dataset import Dataset
from .entity_resolution import resolve_identifiers
from .config import CONTROLLER_TYPES

NODE_TYPES = ["Company", "Person", "Account", "Address", "Contact"]


def build_graph(ds: Dataset, use_controller_edges=True, include_proposed=False, include_accounts=True) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for r in ds.companies.itertuples():
        G.add_node(r.company_id, ntype="Company", sector=r.sector, size=r.declared_size, region=r.region)
    for r in ds.people.itertuples():
        G.add_node(r.person_id, ntype="Person", name=r.name)
    rel = ds.relationships
    if not use_controller_edges:
        rel = rel.iloc[0:0]
    for r in rel.itertuples():
        if r.relationship_type in CONTROLLER_TYPES:
            G.add_edge(r.source_id, r.target_id, etype=r.relationship_type, source=r.source,
                       confidence=float(r.confidence), start=str(r.start_date)[:10], rel_id=r.rel_id)
    contacts, _ = resolve_identifiers(ds.company_contacts, include_proposed=include_proposed)
    for r in contacts.itertuples():
        a, p = f"ADDR::{r.addr_key[:40]}", f"PH::{r.phone_key}"
        G.add_node(a, ntype="Address", label=r.raw_address)
        G.add_node(p, ntype="Contact", label=r.raw_phone)
        G.add_edge(r.company_id, a, etype="REGISTERED_AT", source=r.record_source, confidence=1.0)
        G.add_edge(r.company_id, p, etype="HAS_CONTACT", source=r.record_source, confidence=1.0)
    agg = ds.invoices.groupby(["issuer_id", "receiver_id"])["amount"].agg(n="size", total="sum").reset_index()
    for r in agg.itertuples():
        G.add_edge(r.issuer_id, r.receiver_id, etype="ISSUES_INVOICE_TO", n=int(r.n), total=float(r.total),
                   source="invoice_ledger", confidence=1.0)
    if include_accounts:
        for r in ds.accounts.itertuples():
            G.add_node(r.account_id, ntype="Account")
            G.add_edge(r.linked_entity_id, r.account_id, etype="HAS_ACCOUNT", source="bank_record", confidence=1.0)
        am = ds.accounts.set_index("account_id")["linked_entity_id"]
        tx = ds.transactions
        tagg = tx.groupby(["sender_account", "receiver_account"])["amount"].agg(n="size", total="sum").reset_index()
        for r in tagg.itertuples():
            G.add_edge(r.sender_account, r.receiver_account, etype="TRANSFERS_TO", n=int(r.n),
                       total=float(r.total), source="bank_ledger", confidence=1.0)
    return G


def controller_linked(G, company_id, max_hops=2):
    """Controller-aware traversal: companies reachable through observed OWNS/DIRECTS edges."""
    seen, frontier = {company_id}, {company_id}
    for _ in range(max_hops):
        nxt = set()
        for c in frontier:
            for p, _, d in G.in_edges(c, data=True):
                if d.get("etype") in CONTROLLER_TYPES:
                    for _, c2, d2 in G.out_edges(p, data=True):
                        if d2.get("etype") in CONTROLLER_TYPES and c2 not in seen:
                            nxt.add(c2)
        seen |= nxt
        frontier = nxt
    return sorted(seen - {company_id})


def ego_subgraph(G, company_id, hops=1, max_nodes=60,
                 etypes=("OWNS", "DIRECTS", "ISSUES_INVOICE_TO", "REGISTERED_AT", "HAS_CONTACT")):
    """Neighbourhood of a company over the chosen edge types, capped for readability."""
    H = nx.MultiDiGraph()
    keep = {company_id}
    frontier = {company_id}
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            for u, v, d in list(G.in_edges(n, data=True)) + list(G.out_edges(n, data=True)):
                if d.get("etype") in etypes:
                    nxt |= {u, v}
        # shared address/contact nodes pull in their other tenants
        for n in list(nxt):
            if G.nodes[n].get("ntype") in ("Address", "Contact"):
                nxt |= {u for u, _, d in G.in_edges(n, data=True)}
        keep |= nxt
        frontier = nxt
        if len(keep) > max_nodes:
            break
    keep = list(keep)[:max_nodes] if len(keep) > max_nodes else keep
    keep = set(keep) | {company_id}
    for u, v, d in G.edges(data=True):
        if u in keep and v in keep and d.get("etype") in etypes:
            H.add_edge(u, v, **d)
    for n in keep:
        if n in G:
            H.add_node(n, **G.nodes[n])
    return H


IDENTITY_TYPES = ("OWNS", "DIRECTS", "REGISTERED_AT", "HAS_CONTACT")


def network_view(G, company_id, hops=1, n_partners=5, identity_types=IDENTITY_TYPES, max_nodes=45):
    """A readable neighbourhood: identity links (owner, director, shared address/phone) plus only the
    `n_partners` largest trading partners. Single-use addresses/phones and unshared owners are dropped
    because they add clutter, not information."""
    H = ego_subgraph(G, company_id, hops=hops, max_nodes=max_nodes, etypes=tuple(identity_types))
    for n in list(H.nodes):
        t = H.nodes[n].get("ntype")
        if t in ("Address", "Contact"):
            if sum(1 for _ in H.in_edges(n)) < 2:
                H.remove_node(n)
        elif t == "Person":
            comps = {v for _, v in H.out_edges(n) if H.nodes[v].get("ntype") == "Company"}
            if company_id not in comps and len(comps) < 2:
                H.remove_node(n)
    for n in list(H.nodes):                      # companies that only hung off removed nodes
        if n != company_id and H.nodes[n].get("ntype") == "Company" and H.degree(n) == 0:
            H.remove_node(n)
    if n_partners > 0:
        cand = {}
        for u, v, d in G.out_edges(company_id, data=True):
            if d.get("etype") == "ISSUES_INVOICE_TO":
                cand[v] = cand.get(v, 0) + d["total"]
        for u, v, d in G.in_edges(company_id, data=True):
            if d.get("etype") == "ISSUES_INVOICE_TO":
                cand[u] = cand.get(u, 0) + d["total"]
        for p in sorted(cand, key=cand.get, reverse=True)[:n_partners]:
            if p not in H:
                H.add_node(p, **G.nodes[p])
        keep = [n for n, d in H.nodes(data=True) if d.get("ntype") == "Company"]
        for u in keep:                            # invoice links among the companies on screen
            for v in keep:
                if u != v and G.has_edge(u, v):
                    for _, d in G[u][v].items():
                        if d.get("etype") == "ISSUES_INVOICE_TO":
                            H.add_edge(u, v, **d)
    return H



def layout(H, seed=1):
    """Spread-out layout: identity links pull nodes close, trading links are looser."""
    import networkx as nx
    U = nx.Graph()
    U.add_nodes_from(H.nodes)
    for u, v, d in H.edges(data=True):
        dist = 1.0 if d.get("etype") in IDENTITY_TYPES else 2.0
        if not U.has_edge(u, v) or U[u][v]["dist"] > dist:
            U.add_edge(u, v, dist=dist)
    try:
        return nx.kamada_kawai_layout(U, weight="dist")
    except Exception:
        return nx.spring_layout(U, seed=seed, k=1.5 / max(len(U) ** 0.5, 1))


def layout3d(H, seed=1):
    """3D version of `layout` (identity links pull close, trading links looser)."""
    import networkx as nx
    U = nx.Graph()
    U.add_nodes_from(H.nodes)
    for u, v, d in H.edges(data=True):
        dist = 1.0 if d.get("etype") in IDENTITY_TYPES else 2.0
        if not U.has_edge(u, v) or U[u][v]["dist"] > dist:
            U.add_edge(u, v, dist=dist)
    try:
        return nx.kamada_kawai_layout(U, weight="dist", dim=3)
    except Exception:
        return nx.spring_layout(U, dim=3, seed=seed, k=1.5 / max(len(U) ** 0.5, 1))


def tree_view(H, company_id, score_of=None, keep=None, max_children=12):
    """Turn a neighbourhood into a left-to-right tree.

    root (this company) -> branches (an owner/director, a shared address, a shared phone, the largest
    trading partners) -> companies reached through that branch. A company reachable several ways appears
    under each branch; `also_under` says how many other branches it appears under.

    Returns (nodes, edges). Node: id, kind, label, company, x, y, depth, etype, hover_extra, also_under.
    Edge: (parent_id, child_id, etype).
    """
    score_of = score_of or (lambda c: 0.0)
    keep = keep or (lambda c: True)
    branches = []                                             # (kind, label, etype, [(company, extra)])
    for n, d in H.nodes(data=True):
        t = d.get("ntype")
        if t == "Person":
            links = {}
            for _, v, e in H.out_edges(n, data=True):
                if e["etype"] in CONTROLLER_TYPES and H.nodes[v].get("ntype") == "Company":
                    links.setdefault(v, set()).add(e["etype"])
            if company_id not in links:
                continue
            roles = links[company_id]
            role = "Owner and director" if len(roles) == 2 else ("Owner" if "OWNS" in roles else "Director")
            kids = [(v, "") for v in links if v != company_id]
            branches.append(("person", f"{role}: {d.get('name', n)}", "OWNS" if "OWNS" in roles else "DIRECTS", kids, 0))
        elif t in ("Address", "Contact"):
            comps = [u for u, _, e in H.in_edges(n, data=True) if e["etype"] in ("REGISTERED_AT", "HAS_CONTACT")]
            if company_id not in comps or len(comps) < 2:
                continue
            kind, etype, what = ("address", "REGISTERED_AT", "address") if t == "Address" else ("phone", "HAS_CONTACT", "phone number")
            branches.append((kind, f"Same {what} ({len(comps)} companies)", etype, [(u, "") for u in comps if u != company_id],
                             1 if kind == "address" else 2))
    trade = {}
    for u, v, e in H.edges(data=True):
        if e["etype"] == "ISSUES_INVOICE_TO" and company_id in (u, v):
            other = v if u == company_id else u
            rec = trade.setdefault(other, {"sent": 0.0, "received": 0.0})
            rec["sent" if u == company_id else "received"] += float(e.get("total", 0))
    if trade:
        kids = sorted(trade, key=lambda x: -(trade[x]["sent"] + trade[x]["received"]))
        branches.append(("trade", f"Trading partners ({len(kids)})", "ISSUES_INVOICE_TO",
                         [(k, f"Invoiced to them {trade[k]['sent']:,.0f}; from them {trade[k]['received']:,.0f}") for k in kids], 3))
    branches.sort(key=lambda b: (b[4], b[1]))

    appear = {}
    for _, _, _, kids, _ in branches:
        for k, _ in kids:
            appear[k] = appear.get(k, 0) + 1

    nodes, edges, y = [], [], 0.0
    X = {0: 0.0, 1: 1.35, 2: 2.7}
    branch_ys = []
    for bi, (kind, label, etype, kids, _) in enumerate(branches):
        if kind != "trade":
            kids = sorted(kids, key=lambda kv: -score_of(kv[0]))
        kids = [kv for kv in kids if keep(kv[0])]
        shown, extra = kids[:max_children], max(0, len(kids) - max_children)
        bid = f"b{bi}"
        child_ys = []
        for k, ex in shown:
            nodes.append(dict(id=f"{bid}:{k}", kind="company", label=None, company=k, x=X[2], y=y, depth=2, etype=etype,
                              hover_extra=ex, also_under=appear.get(k, 1) - 1))
            edges.append((bid, f"{bid}:{k}", etype))
            child_ys.append(y)
            y -= 1.0
        if extra:
            nodes.append(dict(id=f"{bid}:more", kind="more", label=f"and {extra} more", company=None, x=X[2], y=y, depth=2,
                              etype=etype, hover_extra="", also_under=0))
            edges.append((bid, f"{bid}:more", etype))
            child_ys.append(y)
            y -= 1.0
        if not child_ys:
            child_ys = [y]
            y -= 1.0
        by = sum(child_ys) / len(child_ys)
        nodes.append(dict(id=bid, kind=kind, label=label, company=None, x=X[1], y=by, depth=1, etype=etype, hover_extra="", also_under=0))
        branch_ys.append(by)
        y -= 0.5
    ry = sum(branch_ys) / len(branch_ys) if branch_ys else 0.0
    nodes.append(dict(id="root", kind="root", label=None, company=company_id, x=X[0], y=ry, depth=0, etype=None, hover_extra="", also_under=0))
    for bi in range(len(branches)):
        edges.append(("root", f"b{bi}", branches[bi][2]))
    return nodes, edges
