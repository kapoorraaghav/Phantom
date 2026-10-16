"""Container for the synthetic tables, with parquet save/load."""
import json
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

TABLES = ["companies", "people", "accounts", "relationships", "company_contacts",
          "invoices", "transactions", "ground_truth", "true_controller_edges"]
# Tables that hold labels / generator internals. Feature code must never read these.
LABEL_TABLES = {"ground_truth", "true_controller_edges"}


@dataclass
class Dataset:
    companies: pd.DataFrame
    people: pd.DataFrame
    accounts: pd.DataFrame
    relationships: pd.DataFrame
    company_contacts: pd.DataFrame
    invoices: pd.DataFrame
    transactions: pd.DataFrame
    ground_truth: pd.DataFrame
    true_controller_edges: pd.DataFrame
    meta: dict

    def save(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        for t in TABLES:
            getattr(self, t).to_parquet(p / f"{t}.parquet", index=False)
        (p / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path):
        p = Path(path)
        kw = {t: pd.read_parquet(p / f"{t}.parquet") for t in TABLES}
        kw["meta"] = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        return cls(**kw)

    @property
    def horizon(self) -> int:
        return int(self.meta["config"]["horizon_days"])
