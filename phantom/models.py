"""Models: rules, Isolation Forest, XGBoost, and a leakage-safe logistic-regression fusion.

Fusion is fitted on OUT-OF-FOLD base-model scores (grouped by ring), never on scores the base
models produced for their own training rows.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from .features import BEHAVIOUR_COLS, GRAPH_COLS, EVIDENCE_COLS


def rule_score(F: pd.DataFrame) -> np.ndarray:
    """Transparent rule baseline: weighted count of red flags, scaled to 0..1."""
    s = np.zeros(len(F))
    s += 1.5 * (F.tight_cycles > 0)
    s += 1.0 * ((F.round_share_out > 0.3) & (F.n_inv_out >= 5))
    s += 1.0 * ((F.median_dwell_days < 3) & (F.flow_balance > 0.85) & (F.n_inv_in >= 5))
    s += 1.0 * (F.ctrl_max_reach >= 3)
    s += 1.0 * ((F.idcluster_size >= 2) & (F.idcluster_size <= 10) & (F.idcluster_internal_share > 0.2))
    s += 1.0 * ((F.max_week_share > 0.30) & (F.dormancy_frac > 0.5))
    s += 1.0 * (F.mismatch_share_out > 0.30)
    return s / 7.5


def _xgb(y, seed):
    pos = max(int(y.sum()), 1)
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=min((len(y) - pos) / pos, 50.0),
                         eval_metric="logloss", n_jobs=2, random_state=seed, verbosity=0)


def _fit_base(kind, X, y, seed):
    if kind == "xgb":
        return _xgb(y, seed).fit(X, y)
    return IsolationForest(n_estimators=200, random_state=seed).fit(X)


def _score_base(kind, m, X):
    return m.predict_proba(X)[:, 1] if kind == "xgb" else -m.score_samples(X)


class PhantomModel:
    BASES = [("xgb_behaviour", "xgb", BEHAVIOUR_COLS),
             ("if_behaviour", "if", BEHAVIOUR_COLS),
             ("xgb_graph", "xgb", GRAPH_COLS)]
    FUSIONS = {"fusion_behaviour": (["xgb_behaviour", "if_behaviour"], False),
               "fusion_graph": (["xgb_graph"], True),
               "fusion_full": (["xgb_behaviour", "if_behaviour", "xgb_graph"], True)}

    def __init__(self, seed=0, n_splits=5):
        self.seed, self.n_splits = seed, n_splits

    def fit(self, F: pd.DataFrame, y, groups):
        y = np.asarray(y)
        groups = np.asarray(groups)
        k = min(self.n_splits, len(set(groups)))
        oof = pd.DataFrame(index=F.index, columns=[b[0] for b in self.BASES], dtype=float)
        for tr, va in GroupKFold(n_splits=k).split(F, y, groups):
            for name, kind, cols in self.BASES:
                m = _fit_base(kind, F.iloc[tr][cols], y[tr], self.seed)
                oof.iloc[va, oof.columns.get_loc(name)] = _score_base(kind, m, F.iloc[va][cols])
        self.base_models = {name: (kind, cols, _fit_base(kind, F[cols], y, self.seed))
                            for name, kind, cols in self.BASES}
        self.fusers = {}
        for fname, (bases, use_ev) in self.FUSIONS.items():
            X = self._fusion_X(oof, F, bases, use_ev)
            lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
            self.fusers[fname] = lr.fit(X, y)
        self.oof_ = oof
        return self

    @staticmethod
    def _fusion_X(S, F, bases, use_ev):
        X = S[bases].copy()
        if use_ev:
            ev = F[EVIDENCE_COLS].copy()
            ev["ev_recency_days"] = np.log1p(ev["ev_recency_days"])
            X = pd.concat([X, ev], axis=1)
        return X

    def predict(self, F: pd.DataFrame) -> pd.DataFrame:
        S = pd.DataFrame(index=F.index)
        for name, (kind, cols, m) in self.base_models.items():
            S[name] = _score_base(kind, m, F[cols])
        out = S.copy()
        out["rules"] = rule_score(F)
        for fname, (bases, use_ev) in self.FUSIONS.items():
            out[fname] = self.fusers[fname].predict_proba(self._fusion_X(S, F, bases, use_ev))[:, 1]
        ranks = np.column_stack([rankdata(S[b]) / len(S) for b in self.FUSIONS["fusion_full"][0]])
        out["fusion_full_uncalibrated"] = ranks.mean(1)     # rank-average, no learned weights/calibration
        return out
