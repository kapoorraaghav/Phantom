"""PHANTOM case desk. Run: streamlit run app.py"""
import html
import json
import re
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from phantom.dataset import Dataset
from phantom.features import ALL_COLS
from phantom.graph import build_graph, network_view, tree_view, IDENTITY_TYPES
from phantom.entity_resolution import resolve_identifiers
from phantom.labels import TIERS, EDGE_LABELS, FEATURE_LABELS, CODE_INFO

st.set_page_config(page_title="PHANTOM", layout="wide", initial_sidebar_state="collapsed")

# brand palette: Black, Gray, Dark Green, Green, Spring Green, White
P0, P1, P2, P3, P4, P5 = "#000000", "#202020", "#16302b", "#02674f", "#05dc80", "#dce4e7"
MUTE = "#97a5a5"                       # secondary text: White at ~65% over the dark greens
GRID = "rgba(220,228,231,.14)"         # chart gridlines
INK, SLATE, RULE, DESK, SHEET = P5, MUTE, P3, P0, P2       # text, secondary text, lines, page, case sheet
STEEL, SIGNAL, AMBER, BOARD = P4, P4, P5, P0             # accent, high (brightest), review, evidence board
FONT = "Manrope, system-ui, sans-serif"
SERIF = "Newsreader, Georgia, serif"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap');
/* text only: never touch the icon font (that caused the overlapping 'arrow_right' text) */
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp summary,
.stApp [data-baseweb], .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {font-family: @FONT@;}
.stApp [data-testid="stIconMaterial"] {font-family: "Material Symbols Rounded" !important;}
.stApp {background: radial-gradient(1100px 520px at 7% -9%, rgba(5,220,126,.22), rgba(2,103,79,.10) 42%, rgba(0,0,0,0) 70%), @P0@; color: @P5@;
        font-variant-numeric: tabular-nums;}
.block-container {padding-top: 1.6rem; max-width: 1340px;}
/* Streamlit's header is laid over the page and swallows clicks (it hid the View / Guide buttons): remove it entirely */
.stAppDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"], header[data-testid="stHeader"] {display: none !important;}
h1, h2, h3, .serif {font-family: @SERIF@ !important; font-weight: 500; letter-spacing: -.005em;}
.wordmark {font-family: @SERIF@; font-size: 1.9rem; font-weight: 500; line-height: 1.05; letter-spacing: .01em; color: @P4@;}
.tagline {color: @MUTE@; font-size: .86rem;}
.notice {color: @MUTE@; font-size: .82rem; margin: .1rem 0 .8rem 0; max-width: 80ch; opacity: .9;}
/* the case sheet and the rail echo the stacked swatches: layers peek out above them */
.st-key-rail {background: @P1@; border: 1px solid rgba(220,228,231,.10); border-radius: 24px; padding: 1rem 1rem .6rem 1rem;
              box-shadow: 0 -7px 0 -3px @P2@, 0 28px 50px -30px rgba(0,0,0,.9);}
.st-key-sheet, .st-key-guide {background: @P2@; border: 1px solid rgba(5,220,126,.22); border-radius: 28px; padding: 1.5rem 1.8rem 1.3rem 1.8rem;
               box-shadow: 0 -8px 0 -4px @P3@, 0 -16px 0 -8px @P1@, 0 34px 60px -34px rgba(0,0,0,.9), inset 0 1px 0 rgba(220,228,231,.10);}
.st-key-sheet {min-height: 720px;}
.st-key-guide {max-width: 920px;}
.st-key-board {background: @P0@; border: 1px solid @P3@; border-radius: 22px; padding: .5rem;}
.case-name {font-family: @SERIF@; font-size: 2.3rem; font-weight: 500; line-height: 1.1; letter-spacing: -.012em; margin: 0;}
.case-meta {color: @MUTE@; font-size: .93rem; margin-top: .35rem;}
.status {margin-top: .8rem; font-size: 1rem;}
.status b {color: @P5@;}
.why {margin-top: .4rem; max-width: 80ch; line-height: 1.55; color: @P5@;}
.qbar {display: flex; height: 14px; border-radius: 999px; overflow: hidden; background: @P3@; animation: wipe 1s ease-out both;}
.qbar span {display: block; height: 100%;}
.hatch {background: repeating-linear-gradient(135deg, @P4@ 0 3px, @P2@ 3px 7px);}
.qlegend {font-size: .88rem; margin: .5rem 0 1rem 0; color: @P5@;}
.qlegend b {font-weight: 600;}
@keyframes wipe {from {clip-path: inset(0 100% 0 0);} to {clip-path: inset(0 0 0 0);}}
.reason-title {font-family: @SERIF@; font-size: 1.3rem; font-weight: 500; margin: 1rem 0 .15rem 0;}
.reason p {margin: .15rem 0; max-width: 78ch; line-height: 1.55;}
.reason .muted {color: @MUTE@;}
.keypanel {color: @P5@; font-size: .86rem; padding: .7rem .6rem .4rem .5rem;}
.keyhead {font-weight: 600; margin-bottom: .35rem; font-family: @SERIF@; font-size: 1.05rem;}
.keyrow {display: flex; align-items: center; gap: .6rem; margin: .34rem 0; line-height: 1.25;}
.kg {display: inline-block; width: 1.1rem; text-align: center; font-size: 1.05rem;}
/* pill controls, like the swatches */
[data-testid="stBaseButton-segmented_control"], [data-testid="stBaseButton-pills"] {border-radius: 999px; border: 1px solid @P3@; background: @P1@; color: @P5@;}
[data-testid="stBaseButton-segmented_controlActive"], [data-testid="stBaseButton-pillsActive"] {border-radius: 999px; background: @P4@ !important; color: @P0@ !important; border: 1px solid @P4@; font-weight: 600;}
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {border-radius: 999px; background: @P4@; color: @P0@; border: none; font-weight: 600;}
[data-baseweb="input"], [data-baseweb="select"] > div {border-radius: 999px !important;}
[data-testid="stExpander"] details {border: 1px solid @P3@; border-radius: 18px; background: rgba(0,0,0,.28);}
[data-testid="stDataFrame"] {border-radius: 16px; overflow: hidden; border: 1px solid @P3@;}
:focus-visible {outline: 2px solid @P4@; outline-offset: 2px;}
::selection {background: @P4@; color: @P0@;}
@media (prefers-reduced-motion: reduce) {* {animation: none !important; transition: none !important;}}
</style>
"""
for k_, v_ in {"@P0@": P0, "@P1@": P1, "@P2@": P2, "@P3@": P3, "@P4@": P4, "@P5@": P5, "@MUTE@": MUTE, "@FONT@": FONT, "@SERIF@": SERIF}.items():
    CSS = CSS.replace(k_, v_)
st.markdown(CSS, unsafe_allow_html=True)

E = html.escape
GLYPH = {"High": "■", "Review": "▲", "Needs more data": "○", "Low": "□"}
VIEWS = ["Cases", "Suspected groups", "Guide"]
SECTIONS = ["Network", "Reasons", "Profile", "Payments"]


def style_fig(fig, dark=False):
    bg = BOARD if dark else "rgba(0,0,0,0)"
    fig.update_layout(font=dict(family=FONT, color=P5, size=13), paper_bgcolor=bg, plot_bgcolor=bg)
    return fig


# ------------------------------------------------------------------ data
RUN_NAMES = {"demo": "Synthetic data", "demo_aml": "IBM AML-Data hybrid", "blind": "Blind test, unseen seed"}
found = [d for d in sorted(Path("outputs").glob("*")) if (d / "scores.parquet").exists()]
top1, top2, top3 = st.columns([1.5, 2.2, 1.3], vertical_alignment="center")
top1.markdown('<div class="wordmark">PHANTOM</div><div class="tagline">Shell company network review</div>', unsafe_allow_html=True)
if found:
    run_dir = top3.selectbox("Dataset", found, format_func=lambda d: RUN_NAMES.get(d.name, d.name), label_visibility="collapsed")
else:
    run_dir = Path(top3.text_input("Run folder", "outputs/demo", label_visibility="collapsed"))
if not (run_dir / "scores.parquet").exists():
    st.error(f"No results in `{run_dir}`. Run `python -m phantom.pipeline all` first.")
    st.stop()


@st.cache_data
def load_run(d):
    d = Path(d)
    return (pd.read_parquet(d / "scores.parquet").set_index("company_id"),
            pd.read_parquet(d / "features.parquet").set_index("company_id"),
            json.loads((d / "reasons.json").read_text(encoding="utf-8")),
            json.loads((d / "clusters.json").read_text(encoding="utf-8")))


@st.cache_resource
def load_ds(d):
    ds = Dataset.load((Path(d) / "dataset_path.txt").read_text(encoding="utf-8").strip())
    return ds, build_graph(ds)


scores, feats, reasons, clusters = load_run(str(run_dir))
ds, G = load_ds(str(run_dir))
NAME = ds.companies.set_index("company_id")["name"].to_dict()
PERSON = ds.people.set_index("person_id")["name"].to_dict()
ranked = scores.sort_values("fusion_full", ascending=False)


def nm(c):
    return NAME.get(c, c)


def pretty(text):
    text = re.sub(r"\bC\d{5,6}\b", lambda m: nm(m.group(0)), str(text))
    return re.sub(r"\bP\d{5}\b", lambda m: PERSON.get(m.group(0), m.group(0)), text)


def tl(t):
    return TIERS[t][0]


def open_case(company, section="Network"):
    st.session_state.company, st.session_state.section, st.session_state.view = company, section, "Cases"


for k, v in (("view", VIEWS[0]), ("section", SECTIONS[0])):
    if st.session_state.get(k) not in (VIEWS if k == "view" else SECTIONS):
        st.session_state[k] = v
if st.session_state.get("company") not in scores.index:
    st.session_state.company = ranked.index[0]

view = top2.segmented_control("View", VIEWS, key="view", label_visibility="collapsed") or VIEWS[0]
st.markdown('<div class="notice">Synthetic demonstration data. Every company and link is fictional, and a match with a real business is coincidence. '
            'Scores set the order of human review; they are not findings of wrongdoing.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ cases
if view == "Cases":
    counts = scores.tier.value_counts()
    n = len(scores)
    segs = [("High", P4, ""), ("Review", P5, ""), ("Needs more data", "", "hatch"), ("Low", P3, "")]
    bar = "".join(f'<span class="{cls}" style="width:{100 * counts.get(t, 0) / n:.2f}%;background:{col}"></span>' if col else
                  f'<span class="{cls}" style="width:{100 * counts.get(t, 0) / n:.2f}%"></span>' for t, col, cls in segs)
    legend = "&nbsp;&nbsp;&nbsp;&nbsp;".join(f'<span style="color:{P4 if t == "High" else (P5 if t == "Review" else MUTE)}">{GLYPH[t]}</span> <b>{counts.get(t, 0):,}</b> {E(tl(t).lower())}' for t, _, _ in segs)
    st.markdown(f'<div class="qbar">{bar}</div><div class="qlegend">{legend}</div>', unsafe_allow_html=True)

    rail_col, sheet_col = st.columns([1, 2.7], gap="medium")
    with rail_col.container(key="rail"):
        sel = st.pills("Show", list(TIERS), selection_mode="multi", default=["High", "Review"], key="tier_filter",
                       format_func=lambda t: f"{GLYPH[t]} {tl(t)} ({counts.get(t, 0):,})", label_visibility="collapsed")
        search = st.text_input("Search", placeholder="Search company name", label_visibility="collapsed")
        q = ranked[ranked.tier.isin(sel or list(TIERS))]
        if search:
            q = q[[search.lower() in nm(i).lower() for i in q.index]]
        q = q.head(400)
        st.session_state["_queue_ids"] = list(q.index)
        view_df = pd.DataFrame({"Company": [f"{GLYPH[t]}  {nm(i)}" for i, t in zip(q.index, q.tier)],
                                "Score": q.fusion_full.round(2).values})

        def pick_row():
            rows = st.session_state["queue_table"]["selection"]["rows"]
            if rows:
                st.session_state.company = st.session_state["_queue_ids"][rows[0]]

        st.dataframe(view_df, hide_index=True, height=560, key="queue_table", on_select=pick_row, selection_mode="single-row",
                     width="stretch", column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0,
                                                                                            format="%.2f", width="small")})
        st.caption(f"{len(view_df):,} shown, highest score first. Click a row to open it.")

    c = st.session_state.company
    t = scores.loc[c, "tier"]
    comp = ds.companies.set_index("company_id").loc[c]
    rc = reasons.get(c)
    if rc is None:
        from phantom.explain import reason_codes
        rc = reason_codes(ds, feats, c)
    with sheet_col.container(key="sheet"):
        st.markdown(f'<div class="case-name">{E(nm(c))}</div>'
                    f'<div class="case-meta">Ref {E(c)}. {E(comp.sector.replace("_", " ").capitalize())} company, {E(comp.declared_size)} size, '
                    f'{E(comp.region)}, registered {str(comp.registration_date)[:10]}.</div>'
                    f'<div class="status">{GLYPH[t]} <b>{E(tl(t))}</b>. Score {scores.loc[c, "fusion_full"]:.2f}, '
                    f'rank {int(scores.loc[c, "rank"])} of {len(scores):,}. {E(TIERS[t][1])}</div>', unsafe_allow_html=True)
        if rc:
            st.markdown('<div class="why">Flagged for: ' + "; ".join(E(CODE_INFO[r["code"]][0].lower()) for r in rc) + ".</div>", unsafe_allow_html=True)
        st.write("")
        section = st.segmented_control("Section", SECTIONS, key="section", label_visibility="collapsed") or SECTIONS[0]

        if section == "Network":
            st.caption("Read left to right: this company, how others are connected to it, then who they are. "
                       "A company that appears under several branches is linked in several ways.")
            with st.expander("Adjust the view"):
                a2, a3 = st.columns([1, 2])
                n_partners = a2.slider("Trading partners", 0, 15, 5, help="Only the largest by invoice value are shown.")
                ident = a3.multiselect("Links", list(IDENTITY_TYPES), default=list(IDENTITY_TYPES), format_func=lambda e: EDGE_LABELS[e])
                only_flagged = st.checkbox("Hide companies that are not flagged", value=False)
            H = network_view(G, c, hops=1, n_partners=n_partners, identity_types=tuple(ident))
            flagged_set = {x for x in scores.index if scores.loc[x, "tier"] in ("High", "Review")}
            nodes, edges = tree_view(H, c, score_of=lambda x: float(scores.loc[x, "fusion_full"]) if x in scores.index else 0.0,
                                     keep=(lambda x: x in flagged_set) if only_flagged else None)
            if len(nodes) <= 1:
                st.info("No links of the selected types. Add trading partners or link types under 'Adjust the view'.")
            else:
                byid = {n_["id"]: n_ for n_ in nodes}
                estyle = {"OWNS": (3.5, "solid", P5), "DIRECTS": (3.5, "dash", P5), "REGISTERED_AT": (3, "solid", P4),
                          "HAS_CONTACT": (3, "dot", P4), "ISSUES_INVOICE_TO": (1.8, "solid", "rgba(5,220,126,0.45)")}
                fig = go.Figure()
                for et, (w, dash, col) in estyle.items():
                    xs, ys = [], []
                    for p_, ch, e_ in edges:
                        if e_ == et:
                            P, C_ = byid[p_], byid[ch]
                            mx = (P["x"] + C_["x"]) / 2
                            xs += [P["x"], mx, mx, C_["x"], None]
                            ys += [P["y"], P["y"], C_["y"], C_["y"], None]
                    if xs:
                        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                                                 line=dict(width=w, dash=dash, color=col, shape="linear")))

                def short(t_, n_=28):
                    return t_ if len(t_) <= n_ else t_[: n_ - 1] + "…"

                groups = {"root": ("circle", 22), "company": ("circle", 11), "more": ("circle-open", 9), "person": ("diamond", 14),
                          "address": ("square", 13), "phone": ("cross", 14), "trade": ("hexagon", 14)}
                for kind, (symbol, size) in groups.items():
                    ns = [n_ for n_ in nodes if n_["kind"] == kind]
                    if not ns:
                        continue
                    cols, rims, widths, texts, hovers = [], [], [], [], []
                    for n_ in ns:
                        cid_ = n_["company"]
                        is_flag = cid_ in flagged_set and kind == "company"
                        if kind == "root":
                            cols.append(P5); rims.append(P4); widths.append(4)
                        elif is_flag:
                            cols.append(P4); rims.append(P5); widths.append(2.5)
                        elif kind == "company":
                            cols.append(P2); rims.append(P4); widths.append(1.5)
                        elif kind == "person":
                            cols.append(P5); rims.append(P0); widths.append(1.5)
                        elif kind in ("address", "phone"):
                            cols.append(P4); rims.append(P0); widths.append(1.5)
                        else:
                            cols.append(P3); rims.append(P4); widths.append(1.5)
                        if kind in ("root", "company"):
                            texts.append(short(nm(cid_)))
                            hv = f"<b>{E(nm(cid_))}</b><br>{E(tl(scores.loc[cid_, 'tier']))}" if cid_ in scores.index else E(nm(cid_))
                            if n_["also_under"]:
                                hv += f"<br>Also linked through {n_['also_under']} other branch" + ("es" if n_["also_under"] > 1 else "")
                            if n_["hover_extra"]:
                                hv += f"<br>{E(n_['hover_extra'])}"
                            hovers.append(hv)
                        else:
                            texts.append(short(n_["label"], 34))
                            hovers.append(E(n_["label"] or ""))
                    leaf = kind in ("company", "more")
                    fig.add_trace(go.Scatter(
                        x=[n_["x"] for n_ in ns], y=[n_["y"] for n_ in ns], mode="markers+text", showlegend=False, text=texts,
                        textposition="middle right" if leaf else "top center", hovertext=hovers, hoverinfo="text",
                        textfont=dict(size=13 if kind == "root" else 12, color=P5),
                        marker=dict(symbol=symbol, size=size, color=cols, line=dict(width=widths, color=rims))))
                style_fig(fig, dark=True)
                height = int(min(900, max(380, 30 * (max(n_["y"] for n_ in nodes) - min(n_["y"] for n_ in nodes)) + 120)))
                fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), hovermode="closest",
                                  xaxis=dict(visible=False, range=[-0.7, 4.6], fixedrange=True),
                                  yaxis=dict(visible=False, fixedrange=True))
                with st.container(key="board"):
                    chart_col, key_col = st.columns([4.2, 1.35])
                    chart_col.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                    key_col.markdown(f"""
<div class="keypanel">
<div class="keyhead">Key</div>
<div class="keyrow"><span class="kg" style="color:{P5}">●</span> This company</div>
<div class="keyrow"><span class="kg" style="color:{P4}">●</span> Flagged: high priority or worth a look</div>
<div class="keyrow"><span class="kg" style="color:{P4}">○</span> Other company</div>
<div class="keyrow"><span class="kg" style="color:{P5}">◆</span> Owner or director</div>
<div class="keyrow"><span class="kg" style="color:{P4}">■</span> Shared address</div>
<div class="keyrow"><span class="kg" style="color:{P4}">✚</span> Shared phone number</div>
<div class="keyrow"><span class="kg" style="color:{P4}">⬢</span> Trading partners</div>
<div class="keyhead" style="margin-top:1rem">Lines</div>
<div class="keyrow"><svg width="36" height="8"><line x1="0" y1="4" x2="36" y2="4" stroke="{P5}" stroke-width="3.5"/></svg> Owner</div>
<div class="keyrow"><svg width="36" height="8"><line x1="0" y1="4" x2="36" y2="4" stroke="{P5}" stroke-width="3.5" stroke-dasharray="7 4"/></svg> Director</div>
<div class="keyrow"><svg width="36" height="8"><line x1="0" y1="4" x2="36" y2="4" stroke="{P4}" stroke-width="3"/></svg> Same address</div>
<div class="keyrow"><svg width="36" height="8"><line x1="0" y1="4" x2="36" y2="4" stroke="{P4}" stroke-width="3" stroke-dasharray="2 4"/></svg> Same phone</div>
<div class="keyrow"><svg width="36" height="8"><line x1="0" y1="4" x2="36" y2="4" stroke="rgba(5,220,126,0.45)" stroke-width="1.8"/></svg> Trades with</div>
</div>""", unsafe_allow_html=True)
                st.caption("Ownership links are only partly recorded, so a company with no branches is not proof of independence. "
                           "Each branch lists up to 12 companies, highest score first.")
                rows = []
                for br in [n_ for n_ in nodes if n_["depth"] == 1]:
                    for ch in [n_ for n_ in nodes if n_["id"].startswith(br["id"] + ":") and n_["company"]]:
                        rows.append({"id": ch["company"], "Company": nm(ch["company"]), "Connected because": br["label"],
                                     "Priority": tl(scores.loc[ch["company"], "tier"]) if ch["company"] in scores.index else "-"})
                if rows:
                    tbl = pd.DataFrame(rows)
                    st.selectbox("Open a connected company", list(dict.fromkeys(tbl.id)), index=None, placeholder="Choose a company to open its file",
                                 format_func=nm, key="jump",
                                 on_change=lambda: st.session_state.update(company=st.session_state.jump) if st.session_state.get("jump") else None)

        elif section == "Reasons":
            if not rc:
                st.info("No specific warning sign fires for this company.")
            for r in rc:
                title, meaning, caution = CODE_INFO[r["code"]]
                st.markdown(f'<div class="reason"><div class="reason-title">{E(title)}</div><p>{E(pretty(r["text"]))}</p>'
                            f'<p class="muted">{E(meaning)}</p></div>', unsafe_allow_html=True)
                with st.expander("Supporting records and innocent explanations"):
                    for s_ in r["support"]:
                        st.write("- " + pretty(s_))
                    if not r["support"]:
                        st.write("No individual records to list for this signal.")
                    st.write(f"**Could be innocent if:** {caution}")
            st.write("")
            with st.expander("Ownership and contact records on file"):
                rel = ds.relationships[ds.relationships.target_id == c]
                if len(rel):
                    st.dataframe(pd.DataFrame({"Person": [PERSON.get(p, p) for p in rel.source_id],
                                               "Role": ["Owner" if x == "OWNS" else "Director" for x in rel.relationship_type],
                                               "Source": rel.source.str.replace("_", " "), "Confidence": rel.confidence.round(2)}),
                                 width="stretch", hide_index=True)
                else:
                    st.write("No ownership or director records are available for this company.")
                contacts, _ = resolve_identifiers(ds.company_contacts)
                row = contacts[contacts.company_id == c].iloc[0]
                st.write(f"**Registered address:** {row.raw_address}.  **Phone:** {row.raw_phone}.")
                conf = int(feats.loc[c, "ev_conflicts"])
                st.caption(f"{int(feats.loc[c, 'ev_n_links'])} supporting link(s)" + (f", {conf} conflicting owner record(s)" if conf else "")
                           + ". Similar-looking addresses are only proposals and are never merged automatically.")

        elif section == "Profile":
            r = feats.loc[c]
            pct = pd.Series({k: (feats[k] <= r[k]).mean() * 100 for k in ALL_COLS})
            top = pct.sub(50).abs().sort_values(ascending=False).head(7).index
            st.markdown("**What stands out against all companies**")
            fig = go.Figure(go.Bar(x=pct[top], y=[FEATURE_LABELS[k] for k in top], orientation="h", marker_color=STEEL,
                                   text=[f"{r[k]:,.2f}" for k in top], textposition="inside", insidetextanchor="start",
                                   textfont=dict(color=P0)))
            fig.update_layout(xaxis=dict(title="Position among all companies (50 is typical)", range=[0, 100], gridcolor=GRID),
                              yaxis=dict(autorange="reversed"), height=360, margin=dict(l=0, r=10, t=6, b=0))
            fig.add_vline(x=50, line_dash="dot", line_color=SLATE)
            st.plotly_chart(style_fig(fig), width="stretch")
            st.caption("Standing out is not wrongdoing: seasonal, cash-heavy and wholesale businesses also differ from the typical company.")

        else:
            st.caption("Money in and money out for this company. Look for long quiet spells followed by bursts, or money leaving soon after it arrives.")
            acc = ds.accounts[ds.accounts.linked_entity_id == c].account_id
            tx = ds.transactions
            amap = ds.accounts.set_index("account_id").linked_entity_id
            out_ = tx[tx.sender_account.isin(acc)].assign(direction="Money out", other=lambda x: x.receiver_account.map(amap))
            in_ = tx[tx.receiver_account.isin(acc)].assign(direction="Money in", other=lambda x: x.sender_account.map(amap))
            pay = pd.concat([out_, in_]).sort_values("timestamp")
            if pay.empty:
                st.info("No payments recorded for this company.")
            else:
                pay["Other company"] = pay.other.map(nm)
                fig = go.Figure()
                for dname, symb, col in (("Money in", "triangle-down", P5), ("Money out", "triangle-up-open", P4)):
                    s_ = pay[pay.direction == dname]
                    fig.add_trace(go.Scatter(x=s_.timestamp, y=s_.amount, mode="markers", name=dname, hovertext=s_["Other company"],
                                             marker=dict(symbol=symb, size=10, color=col, line=dict(width=2, color=col))))
                fig.update_yaxes(type="log", title="Amount (log scale)", gridcolor=GRID)
                fig.update_xaxes(gridcolor=GRID)
                fig.update_layout(height=380, margin=dict(l=0, r=0, t=6, b=0))
                st.plotly_chart(style_fig(fig), width="stretch")
                wk = pay.assign(week=pay.timestamp.dt.to_period("W").dt.start_time).groupby("week").size().rename("Payments per week")
                st.bar_chart(wk, height=150, color=STEEL)
                with st.expander("All payments"):
                    st.dataframe(pay[["timestamp", "direction", "Other company", "amount"]].rename(
                        columns={"timestamp": "Date", "direction": "Direction", "amount": "Amount"}).tail(300), width="stretch", hide_index=True)

# ------------------------------------------------------------------ groups
elif view == "Suspected groups":
    st.subheader("Suspected groups")
    st.write("Flagged companies that share an owner, address or phone, or trade repeatedly with each other. Each group is a candidate ring for a person to confirm or reject.")
    if not clusters:
        st.info("No groups found.")
    else:
        gdf = pd.DataFrame({"Group": [k["cluster_id"] for k in clusters], "Companies": [k["size"] for k in clusters],
                            "Average score": [round(k["mean_score"], 2) for k in clusters],
                            "Includes": [", ".join(nm(m) for m in k["members"][:3]) + (f" and {k['size'] - 3} more" if k["size"] > 3 else "") for k in clusters]})
        ev = st.dataframe(gdf, hide_index=True, width="stretch", height=min(380, 40 + 35 * len(gdf)), on_select="rerun",
                          selection_mode="single-row", key="groups_table")
        rows = ev.selection.rows if ev and ev.selection else []
        if rows:
            k = clusters[rows[0]]
            st.markdown(f"**Group {k['cluster_id']}: {k['size']} companies**")
            st.dataframe(pd.DataFrame({"Company": [nm(m) for m in k["members"]], "Priority": [tl(scores.loc[m, "tier"]) for m in k["members"]],
                                       "Score": [round(float(scores.loc[m, "fusion_full"]), 2) for m in k["members"]]}),
                         hide_index=True, width="stretch")
            st.markdown("**Why they are grouped**")
            st.dataframe(pd.DataFrame([{"Company A": nm(e["a"]), "Company B": nm(e["b"]), "Linked because": "; ".join(pretty(w) for w in e["why"])}
                                       for e in k["evidence"]]), hide_index=True, width="stretch")
            best = max(k["members"], key=lambda m: scores.loc[m, "fusion_full"])
            st.button(f"Open {nm(best)} in the network", on_click=open_case, args=(best, "Network"))
        else:
            st.caption("Select a group to see its members and the links that join them.")
    st.caption("A controller missing from the records cannot be found.")

# ------------------------------------------------------------------ guide
else:
    with st.container(key="guide"):
        st.subheader("How to read this")
        st.markdown("**Score.** A number from 0 to 1 that sets the order of review. Higher means look sooner. It is not the probability that a company is a shell.")
        st.markdown("**Priority levels.**")
        for t, (label, desc) in TIERS.items():
            st.markdown(f"- {GLYPH[t]} **{label}**: {desc}")
        st.markdown("**What the signals mean.** Each reason says what was found, what it usually indicates, and what innocent explanation to rule out before acting.")
        st.markdown("**Limits.** Ownership links are only partly recorded, so a missing link never shows that a company is independent. Coworking offices, "
                    "business groups and wholesalers can look like shell networks. Similar-looking addresses are only suggested as matches and are never merged automatically.")
