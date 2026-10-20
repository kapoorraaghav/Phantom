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
| `phantom/amlworld.py` | **hybrid loader**: IBM AML-Data transfers + laundering labels, with our synthetic company / controller / address / invoice layers on top |
| `phantom/names.py`, `phantom/labels.py` | fictional readable company names (pure function of seed + sector, label-free); plain-language wording for tiers, reasons and features |
| `app.py`, `.streamlit/config.toml` | case-desk dashboard: work list on the left, selected company's case file on the right (Network as a left-to-right tree with the key beside it, Reasons, Profile, Payments), suspected groups, guide. The config file sets the dark-green theme (#051F20, #0B2B26, #163832, #235347, #8EB69B, #DAF1DE); keep it next to `app.py` |

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

Fusion PR-AUC by setting: ring-grouped 0.83; no controller edges 0.79; fuzzy links merged 0.82; temporal+grouped 0.73; **random split 0.88 (inflated by leakage)**. Hold-out typology: circular 0.94, dormant-then-spike 0.94, pass-through 0.69, **camouflaged 0.47** (the honest weak spot). Blind demo (unseen seed 777): PR-AUC 0.83; candidate clusters recover 13 of 16 rings. Full tables: `outputs/eval/experiments_summary.md`.

## Read these before quoting any number
- The std is large (±0.12–0.27) because there are only ~8 test rings per split. Differences under ~0.1 PR-AUC are not reliable; the controller sweep is non-monotone for the same reason.
- The generator and the features were written by the same person, so results show the pipeline finds *these* typologies, not that it finds real shell companies.
- The XGBoost numbers use synthetic labels; treat them as an upper bound. Rules and Isolation Forest are the more honest baselines.
- Re-run with other seeds (`--seed`) and larger `--n-companies` before putting figures in a deck.

## Using IBM AML-Data (hybrid, Option B)
**Easiest way:** put `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` in `data\aml\`, then run `python -m phantom.pipeline all`.
It detects the files automatically, builds the hybrid dataset next to the pure-synthetic one, evaluates and scores both
(`outputs\eval` / `outputs\demo` = synthetic, `outputs\eval_aml` / `outputs\demo_aml` = hybrid), and the dashboard's
"Dataset" box lets you switch. With no IBM files it simply runs synthetic-only. If the hybrid build fails it prints the
reason and carries on with synthetic. `python -m phantom.pipeline generate` follows the same rule (`--source synthetic|aml` to force).
Use `all --quick` for a fast check.

Manual steps, if you prefer:
The repo https://github.com/IBM/AML-Data holds no data; download `HI-Small_Trans.csv` and `HI-Small_Patterns.txt`
from the Kaggle/Box links in its README (licence: CDLA-Sharing-1.0 for the data).
```bash
python -m phantom.pipeline inspect-aml --trans HI-Small_Trans.csv --patterns HI-Small_Patterns.txt
python -m phantom.pipeline load-aml    --trans HI-Small_Trans.csv --patterns HI-Small_Patterns.txt --out data/aml_hi_small
python -m phantom.pipeline evaluate    --data data/aml_hi_small --out outputs/eval_aml
python -m phantom.pipeline score       --data data/aml_hi_small --out outputs/demo_aml
```
- **From IBM:** accounts, transfers, timestamps, amounts, the laundering flag and the named attempt types (cycle, fan-out, ...). Each attempt (merged if attempts share accounts) is one ring, so splits and typology hold-outs use IBM's own pattern types.
- **Ours (synthetic):** company attributes, hidden controllers shared by each ring (partially observed + noise), shared addresses/phones, and one invoice per transfer. The invoice layer uses the same rule for every account, so it carries no label information; detection must come from IBM's transfer structure plus the controller/identity layer.
- **Real-file check (HI-Small):** 5,078,345 transfers, 5,177 laundering transfers (0.10%), 515,088 accounts, 1-18 Sep 2022, 370 laundering attempts (median 8 accounts). Pattern headers carry parameters (`CYCLE: Max 2 hops`); the loader keeps only the family (cycle, fan-in, fan-out, gather-scatter, scatter-gather, bipartite, stack, random).
- **Prevalence is deliberately inflated** (default ring_fraction 0.07, about 7% of evaluated accounts vs 0.1% in IBM's data) so there are enough rings (~25+) for grouped splits. Compare every metric with the random baseline = prevalence.
- **Which hybrid number is independent?** Only `ring_grouped, TRANSFERS ONLY (no controller/address/phone links)`. The headline hybrid score also uses our synthetic owner/address/phone layer, which marks rings by construction. Quote the transfers-only row as the claim about IBM's own signal.
- **Sampling:** all accounts of sampled attempts + uniformly sampled benign accounts are the evaluated "seeds"; their counterparties ("fringe") give graph context and are never scored. The induced graph is much sparser than IBM's full graph, so results can be optimistic.
- **Status:** written against the format as I understand it (header `Timestamp, From Bank, Account, To Bank, Account, Amount Received, ... Is Laundering`; pattern blocks `BEGIN/END LAUNDERING ATTEMPT - <TYPE>`). It was tested ONLY on a mock file in that format (`tests/mock_aml.py`), not on the real download. Run `inspect-aml` first; if it errors, send the header / first lines of each file.
- The `blind` command is for the pure-synthetic generator only.

## Not implemented (yet)
- AMLSim loader.
- R-GCN (stretch goal; train and test must be separate graphs to avoid message-passing leakage).
- `LINKED_BY_RECORD` edges, MLflow, FastAPI.
