"""Light-weight entity resolution for addresses and contacts.

Policy (from the plan):
  * exact match after normalisation -> link, confidence 1.0
  * fuzzy match -> *proposed* link only, never silently merged
"""
import re
from difflib import SequenceMatcher
import pandas as pd

_TOKEN_MAP = {"rd": "road", "st": "street", "ave": "avenue", "ngr": "nagar", "blk": "block"}
_DROP = {"no", "number"}


def normalise_address(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [_TOKEN_MAP.get(t, t) for t in s.split() if t not in _DROP]
    return " ".join(toks)


def normalise_phone(s: str) -> str:
    digits = re.sub(r"\D", "", str(s))
    return digits[-10:]


def _pin(norm_addr: str) -> str:
    m = re.search(r"(\d{6})\s*$", norm_addr)
    return m.group(1) if m else ""


def propose_fuzzy_address_links(norm_addresses, threshold: float = 0.93) -> pd.DataFrame:
    """Candidate address pairs that are similar but not identical. Blocked by pincode."""
    uniq = sorted(set(norm_addresses))
    blocks = {}
    for a in uniq:
        blocks.setdefault(_pin(a), []).append(a)
    rows = []
    for _, items in blocks.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                r = SequenceMatcher(None, items[i], items[j]).ratio()
                if r >= threshold:
                    rows.append((items[i], items[j], r))
    return pd.DataFrame(rows, columns=["addr_a", "addr_b", "confidence"]).assign(status="proposed")


def resolve_identifiers(contacts: pd.DataFrame, include_proposed: bool = False, threshold: float = 0.93):
    """Return contacts with `addr_key`/`phone_key` columns.

    With include_proposed=True, fuzzy-matched addresses are merged under one key
    (used only for the entity-resolution error sensitivity test).
    """
    out = contacts.copy()
    out["addr_norm"] = out["raw_address"].map(normalise_address)
    out["phone_key"] = out["raw_phone"].map(normalise_phone)
    out["addr_key"] = out["addr_norm"]
    proposals = propose_fuzzy_address_links(out["addr_norm"], threshold)
    if include_proposed and len(proposals):
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in zip(proposals.addr_a, proposals.addr_b):
            parent[find(a)] = find(b)
        out["addr_key"] = out["addr_norm"].map(find)
    return out, proposals
