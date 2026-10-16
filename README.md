# PHANTOM — development prototype

Controller-aware shell-company network detection on **synthetic data**. Nothing here uses real
GST, bank or MCA records, and no number below is real-world performance.

## Quick start
```bash
pip install -r requirements.txt
python -m phantom.pipeline all          # generate -> evaluate -> sweep -> score -> blind (~5 min, CPU)
streamlit run app.py                    # dashboard (reads outputs/demo; sidebar can point at outputs/blind)
pytest -q                               # leakage / sanity tests
```
Individual steps: `generate`, `evaluate`, `sweep`, `score`, `blind` (see `phantom/pipeline.py`).

## Layout
| File | Role |
|---|---|
| `phantom/config.py` | constants, generator config (ring fraction, controller observation rate, noise) |
| `phantom/generator.py` | benign economy + 4 ring typologies + benign look-alikes; writes labels only to `ground_truth` |
| `phantom/entity_resolution.py` | address/phone normalisation; fuzzy matches are *proposed*, never silently merged |
| `phantom/graph.py` | typed, provenance-carrying heterogeneous graph; controller-aware traversal |
| `phantom/features.py` | behaviour, transaction-graph, identity, controller and evidence features (window-aware) |
| `phantom/models.py` | rules, Isolation Forest, XGBoost, logistic-regression fusion on out-of-fold scores |
| `phantom/evaluation.py` | ring-grouped / temporal / random splits, metrics, ablations, controller sweep |
| `phantom/explain.py` | tiers, "Needs more data" rule, reason codes with supporting records, candidate clusters |
| `app.py` | Streamlit dashboard: queue, network explorer, profile, timeline, evidence, case summary |

## Design choices that address the plan review
- **Controller edges are partial and noisy** (default 50% observed, ~15% extra wrong links). `controller_sweep` re-runs detection at 0–100% observation.
- **Label isolation:** `ground_truth` / `true_controller_edges` are read only by splitting and metrics; `tests/test_leakage.py` enforces it.
- **Splits:** headline = ring-grouped; also temporal+grouped, hold-one-typology-out, and a random split labelled *diagnostic only*.
- **Fusion** is fitted on out-of-fold base scores (grouped by ring). `fusion_full_uncalibrated` is a rank-average for comparison.
- **Target label:** entity is positive if it is a member of an injected ring (~2.6% prevalence).
- **Benign look-alikes:** seasonal, cash-heavy, wholesaler firms; coworking sites (many unrelated firms, one address); legitimate multi-company owners; reciprocal trade; near-duplicate addresses.
- **Tiers:** High ≥ 0.50, Review ≥ 0.20 on the fused score; "Needs more data" if < 10 transactions, < 60 days history, or Review-band suspicion with no corroborating link (`explain.assign_tiers`).

## Measured results (this seed, synthetic only)
4,000 companies, 17 rings, prevalence 2.56%. A random ranking gets PR-AUC ≈ 0.026. Ring-grouped splits, 5 repeats, mean ± std:

| variant | PR-AUC | Precision@Npos | Recall@2·Npos | FPR (top 2·Npos flagged) |
|---|---|---|---|---|
| rules | 0.54 ± 0.25 | 0.56 | 0.60 | 0.037 |
| Isolation Forest (behaviour) | 0.42 ± 0.27 | 0.38 | 0.44 | 0.041 |
| XGBoost (behaviour) | 0.65 ± 0.16 | 0.60 | 0.63 | 0.036 |
| XGBoost (graph) | 0.76 ± 0.18 | 0.71 | 0.79 | 0.032 |
| fusion (full) | 0.83 ± 0.12 | 0.81 | 0.84 | 0.030 |
| fusion (rank-average, uncalibrated) | 0.63 ± 0.20 | 0.59 | 0.64 | 0.035 |

Fusion PR-AUC by setting: ring-grouped 0.83; no controller edges 0.79; fuzzy links merged 0.82; temporal+grouped 0.73; **random split 0.88 (inflated by leakage)**. Hold-out typology: circular 0.91, dormant-then-spike 0.94, pass-through 0.71, **camouflaged 0.43** (the honest weak spot). Blind demo (unseen seed 777): PR-AUC 0.83; candidate clusters recover 13 of 16 rings. Full tables: `outputs/eval/experiments_summary.md`.

## Read these before quoting any number
- The std is large (±0.12–0.27) because there are only ~8 test rings per split. Differences under ~0.1 PR-AUC are not reliable; the controller sweep is non-monotone for the same reason.
- The generator and the features were written by the same person, so results show the pipeline finds *these* typologies, not that it finds real shell companies.
- The XGBoost numbers use synthetic labels; treat them as an upper bound. Rules and Isolation Forest are the more honest baselines.
- Re-run with other seeds (`--seed`) and larger `--n-companies` before putting figures in a deck.

## Not implemented (yet)
- AML-Data / AMLSim loader. Plug-in point: map accounts to `accounts.linked_entity_id`, then keep the generator's invoice/controller layers on top.
- R-GCN (stretch goal; train and test must be separate graphs to avoid message-passing leakage).
- `LINKED_BY_RECORD` edges, MLflow, FastAPI.
