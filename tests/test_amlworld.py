"""Loader plumbing test on a MOCK file in the believed AML-Data format (not IBM data)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pytest
from mock_aml import make_mock
from phantom.amlworld import build_dataset, detect_columns, inspect_aml
from phantom.features import build_features
from phantom.evaluation import eval_gt


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    d = tmp_path_factory.mktemp("aml")
    t, p = make_mock(str(d), n_acc=3000, n_tx=20000, n_attempts=40)
    return build_dataset(t, p, n_benign_seeds=400, ring_fraction=0.15, log=lambda *_: None), t, p


def test_detects_duplicate_account_header():
    cm = detect_columns(["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received",
                         "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"])
    assert cm["from_account"] == "Account" and cm["to_account"] == "Account.1"


def test_bad_header_gives_actionable_error():
    with pytest.raises(ValueError, match="Header found"):
        detect_columns(["a", "b", "c"])


def test_only_seeds_are_evaluated_and_rings_are_whole(ds):
    d, _, _ = ds
    gt = d.ground_truth
    assert gt.in_eval.sum() < len(gt)                 # fringe exists and is excluded
    ev = eval_gt(d)
    assert ev.synthetic_label.sum() > 0 and ev.ring_id.notna().sum() > 0
    assert (ev[ev.ring_id.notna()].synthetic_label == 1).all()


def test_invoice_layer_is_label_neutral(ds):
    """Same lag/day rule and item-noise rate for every account: no ring/benign difference."""
    d, _, _ = ds
    assert (d.invoices.day.values == d.transactions.day.values).all()


def test_features_run_and_have_no_labels(ds):
    d, _, _ = ds
    F = build_features(d)
    assert "synthetic_label" not in F.columns and len(F) == len(d.companies)


def test_find_aml_files_prefers_hi_small(tmp_path):
    from phantom.pipeline import find_aml_files
    d = tmp_path / "data" / "aml"
    d.mkdir(parents=True)
    for n in ("LI-Small_Trans.csv", "HI-Small_Trans.csv", "HI-Small_Patterns.txt", "readme.txt"):
        (d / n).write_text("x", encoding="utf-8")
    t, p = find_aml_files(tmp_path)
    assert t.name == "HI-Small_Trans.csv" and p.name == "HI-Small_Patterns.txt"
    assert find_aml_files(tmp_path / "nowhere") == (None, None)


def test_pattern_types_are_collapsed_to_families(tmp_path):
    from phantom.amlworld import parse_patterns
    t, p = make_mock(str(tmp_path), n_acc=500, n_tx=2000, n_attempts=20)
    types = {a["type"] for a in parse_patterns(p)}
    assert types <= {"cycle", "fan-out", "fan-in", "scatter-gather"}, types
    f = tmp_path / "real_style.txt"
    f.write_text("BEGIN LAUNDERING ATTEMPT - GATHER-SCATTER:  Max 3 degree fan-in\n"
                 "2022/09/01 00:20,011,A1,012,B1,10.0,US Dollar,10.0,US Dollar,ACH,1\n"
                 "END LAUNDERING ATTEMPT - GATHER-SCATTER:  Max 3 degree fan-in\n", encoding="utf-8")
    assert [a["type"] for a in parse_patterns(f)] == ["gather-scatter"]
