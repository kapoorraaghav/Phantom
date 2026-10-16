"""PHANTOM investigator dashboard (Streamlit). Run: streamlit run app.py"""
import json
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from phantom.dataset import Dataset
from phantom.features import ALL_COLS
from phantom.graph import build_graph, ego_subgraph
from phantom.entity_resolution import resolve_identifiers

st.set_page_config(page_title="PHANTOM", layout="wide")
st.error("SYNTHETIC DATA — every entity, link and score here is generated. Scores prioritise human review; "
         "they are not findings of wrongdoing.")

run_dir = Path(st.sidebar.text_input("Run folder", "outputs/demo"))
if not (run_dir / "scores.parquet").exists():
    st.warning(f"No run found in {run_dir}. Run `python -m phantom.pipeline score` first.")
    st.stop()


@st.cache_data
def load_run(d):
    d = Path(d)
    scores = pd.read_parquet(d / "scores.parquet").set_index("company_id")
    feats = pd.read_parquet(d / "features.parquet").set_index("company_id")
    reasons = json.loads((d / "reasons.json").read_text(encoding="utf-8"))
    clusters = json.loads((d / "clusters.json").read_text(encoding="utf-8"))
    return scores, feats, reasons, clusters


@st.cache_resource
def load_ds(d):
    ds = Dataset.load((Path(d) / "dataset_path.txt").read_text(encoding="utf-8").strip())
    return ds, build_graph(ds)


scores, feats, reasons, clusters = load_run(str(run_dir))
ds, G = load_ds(str(run_dir))
screen = st.sidebar.radio("Screen", ["1 Investigation queue", "2 Network explorer", "3 Entity profile",
                                      "4 Transaction timeline", "5 Evidence panel", "6 Case summary"])
tiers = ["High", "Review", "Needs more data", "Low"]

if "company" not in st.session_state:
    st.session_state.company = scores.sort_values("fusion_full", ascending=False).index[0]
ranked = scores.sort_values("fusion_full", ascending=False)
st.sidebar.selectbox("Selected company", ranked.index, key="company",
                     format_func=lambda c: f"{c}  ({ranked.loc[c, 'tier']}, {ranked.loc[c, 'fusion_full']:.2f})")
c = st.session_state.company

LINE = {"OWNS": "solid", "DIRECTS": "dash", "ISSUES_INVOICE_TO": "dot", "REGISTERED_AT": "dashdot",
        "HAS_CONTACT": "longdash", "TRANSFERS_TO": "dot"}
SYMBOL = {"Company": "circle", "Person": "diamond", "Address": "square", "Contact": "triangle-up", "Account": "x"}

if screen.startswith("1"):
    st.header("Investigation queue")
    sel = st.multiselect("Priority", tiers, default=["High", "Review", "Needs more data"])
    q = ranked[ranked.tier.isin(sel)][["rank", "tier", "fusion_full", "xgb_behaviour", "xgb_graph", "rules"]]
    q = q.assign(top_reason=[", ".join(r["code"] for r in reasons.get(i, [])[:3]) for i in q.index])
    st.caption("Score = prioritisation signal (fused, calibrated on synthetic validation folds), not a probability of wrongdoing. "
               "'Needs more data' = too little history, or suspicion with no corroborating links.")
    st.dataframe(q.round(3), width="stretch", height=520)
    st.write(f"{len(q)} companies shown of {len(scores)}.")

elif screen.startswith("2"):
    st.header(f"Network explorer — {c}")
    hops = st.slider("Hops", 1, 2, 1)
    types = st.multiselect("Edge types", list(LINE), default=["OWNS", "DIRECTS", "ISSUES_INVOICE_TO", "REGISTERED_AT", "HAS_CONTACT"])
    H = ego_subgraph(G, c, hops=hops, max_nodes=60, etypes=tuple(types))
    if len(H) > 1:
        U = nx.Graph(H)
        pos = nx.spring_layout(U, seed=1, k=0.6)
        fig = go.Figure()
        for et in types:
            xs, ys = [], []
            for u, v, d in H.edges(data=True):
                if d["etype"] == et:
                    xs += [pos[u][0], pos[v][0], None]; ys += [pos[u][1], pos[v][1], None]
            if xs:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=et, line=dict(width=1.4, dash=LINE[et], color="#555")))
        for nt in SYMBOL:
            ns = [n for n, d in H.nodes(data=True) if d.get("ntype") == nt]
            if ns:
                fig.add_trace(go.Scatter(
                    x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers+text", name=nt,
                    text=[n if nt != "Address" and nt != "Contact" else "" for n in ns], textposition="top center",
                    hovertext=[f"{n} | {scores.loc[n, 'tier'] if n in scores.index else ''} | {H.nodes[n].get('label', H.nodes[n].get('name', ''))}" for n in ns],
                    marker=dict(size=[18 if n == c else 11 for n in ns], symbol=SYMBOL[nt],
                                color=["#1f3b73" if n == c else "#9aa5b1" for n in ns], line=dict(width=1, color="#222"))))
        fig.update_layout(height=620, showlegend=True, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("Meaning is carried by marker shape and line style, not colour. Selected company is the large dark marker. "
                   "Controller edges are partial and noisy by design: absence of an edge is not evidence of independence.")
        edges = pd.DataFrame([{"from": u, "to": v, "type": d.get("etype"), "provenance": d.get("source"),
                               "confidence": d.get("confidence"), "n_invoices": d.get("n"), "total_value": d.get("total")}
                              for u, v, d in H.edges(data=True)])
        st.dataframe(edges, width="stretch")
    else:
        st.info("No links of the selected types.")

elif screen.startswith("3"):
    st.header(f"Entity profile — {c}")
    comp = ds.companies.set_index("company_id").loc[c]
    st.write(f"**Declared:** sector `{comp.sector}`, size `{comp.declared_size}`, region `{comp.region}`, registered {str(comp.registration_date)[:10]}")
    r = feats.loc[c]
    pct = pd.Series({k: (feats[k] <= r[k]).mean() * 100 for k in ALL_COLS})
    top = pct.sub(50).abs().sort_values(ascending=False).head(12).index
    fig = go.Figure(go.Bar(x=pct[top], y=top, orientation="h", marker_color="#1f3b73",
                           text=[f"{r[k]:.2f}" for k in top], textposition="outside"))
    fig.update_layout(xaxis_title="percentile among all companies", height=450, yaxis=dict(autorange="reversed"), margin=dict(l=0, r=40, t=10, b=0))
    fig.add_vline(x=50, line_dash="dot")
    st.plotly_chart(fig, width="stretch")
    st.caption("Largest deviations from the population (value shown at bar end). Benign look-alikes (seasonal, cash-heavy, wholesale) also deviate.")
    st.dataframe(scores.loc[[c], ["tier", "fusion_full", "xgb_behaviour", "if_behaviour", "xgb_graph", "rules"]].round(3))

elif screen.startswith("4"):
    st.header(f"Transaction timeline — {c}")
    acc = ds.accounts[ds.accounts.linked_entity_id == c].account_id
    tx = ds.transactions
    amap = ds.accounts.set_index("account_id").linked_entity_id
    out_ = tx[tx.sender_account.isin(acc)].assign(direction="out", counterparty=lambda d: d.receiver_account.map(amap))
    in_ = tx[tx.receiver_account.isin(acc)].assign(direction="in", counterparty=lambda d: d.sender_account.map(amap))
    t = pd.concat([out_, in_]).sort_values("timestamp")
    fig = go.Figure()
    for d, sym in (("in", "triangle-down"), ("out", "triangle-up")):
        s = t[t.direction == d]
        fig.add_trace(go.Scatter(x=s.timestamp, y=s.amount, mode="markers", name=f"payments {d}", marker=dict(symbol=sym, size=8),
                                 hovertext=s.counterparty + " " + s.transaction_id))
    fig.update_yaxes(type="log", title="amount (INR, log)")
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    wk = t.assign(week=t.timestamp.dt.to_period("W").dt.start_time).groupby("week").size()
    st.bar_chart(wk)
    st.dataframe(t[["transaction_id", "timestamp", "direction", "counterparty", "amount", "invoice_id"]].tail(200), width="stretch")

elif screen.startswith("5"):
    st.header(f"Evidence panel — {c}")
    st.write(f"**Tier:** {scores.loc[c, 'tier']} | **fused score:** {scores.loc[c, 'fusion_full']:.3f}")
    rc = reasons.get(c)
    if rc is None:
        from phantom.explain import reason_codes
        rc = reason_codes(ds, feats, c)
    if not rc:
        st.info("No reason codes fire for this company.")
    for r in rc:
        with st.expander(f"{r['code']} — {r['text']}", expanded=True):
            st.json(r["features"])
            for s in r["support"]:
                st.write("• " + s)
    st.subheader("Source records and match confidence")
    rel = ds.relationships[ds.relationships.target_id == c][["source_id", "relationship_type", "source", "confidence", "start_date"]]
    st.dataframe(rel, width="stretch")
    contacts, props = resolve_identifiers(ds.company_contacts)
    st.write("Registered address / phone on file:", contacts[contacts.company_id == c][["raw_address", "raw_phone"]].to_dict("records"))
    st.caption("Fuzzy-matched links are only proposed, never merged. Missing or conflicting evidence: "
               f"{int(feats.loc[c, 'ev_conflicts'])} conflicting owner record(s); {int(feats.loc[c, 'ev_n_links'])} supporting link(s).")

else:
    st.header("Case summary — candidate rings")
    if not clusters:
        st.info("No candidate clusters.")
    for k in clusters[:25]:
        with st.expander(f"{k['cluster_id']}: {k['size']} companies, mean score {k['mean_score']:.2f}"):
            st.write(", ".join(k["members"]))
            ev = pd.DataFrame([dict(a=e["a"], b=e["b"], supporting_link="; ".join(e["why"])) for e in k["evidence"]])
            st.dataframe(ev, width="stretch")
            st.text_area("Reviewer notes", key=f"n_{k['cluster_id']}", placeholder="Reviewer notes (not saved)")
    st.caption("Limitations: clusters link only flagged companies through observed controllers, shared identifiers and repeated invoices. "
               "Hidden controllers absent from the records cannot be found.")
