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
