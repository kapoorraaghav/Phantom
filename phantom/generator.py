"""Synthetic generator: benign economy + injected ring typologies + hard negatives.

Everything that identifies a ring (ring_id, controller, typology, seed) is written ONLY to
`ground_truth` / `true_controller_edges`. The observable tables contain just what an analyst
would plausibly see: records, invoices, payments and (partial, noisy) controller links.
"""
import numpy as np
import pandas as pd
from .config import *
from .dataset import Dataset
from .names import make_company_names

FIRST = ["Rajesh", "Amit", "Sunita", "Priya", "Vikram", "Neha", "Arjun", "Kavita", "Manoj", "Anita",
         "Suresh", "Pooja", "Rahul", "Deepa", "Sanjay", "Meera", "Karan", "Ritu", "Vivek", "Asha"]
LAST = ["Sharma", "Verma", "Gupta", "Singh", "Kumar", "Mehta", "Iyer", "Reddy", "Patel", "Nair",
        "Jain", "Agarwal", "Bose", "Khanna", "Malhotra", "Chopra", "Joshi", "Rao", "Das", "Shah"]
STREETS = ["MG Road", "Nehru Marg", "Station Road", "Gandhi Street", "Park Avenue", "Industrial Area Phase 2",
           "Ring Road", "Market Street", "Lake View Road", "Civil Lines"]
AREAS = ["Sector 5", "Block C", "Nagar East", "Old Town", "Tech Park", "Mill Compound"]
CITY_PIN = {"Delhi": "110", "Maharashtra": "400", "Karnataka": "560",
            "Gujarat": "380", "Tamil Nadu": "600", "Uttar Pradesh": "226"}


def _canon_address(rng, region):
    return dict(num=int(rng.integers(1, 400)), street=str(rng.choice(STREETS)),
                area=str(rng.choice(AREAS)), region=region,
                pin=CITY_PIN[region] + f"{int(rng.integers(0, 100)):03d}")


def _fmt_address(rng, c):
    street = c["street"]
    if rng.random() < 0.4:
        street = street.replace("Road", "Rd.").replace("Street", "St.").replace("Avenue", "Ave")
    prefix = "No. " if rng.random() < 0.3 else ""
    s = f"{prefix}{c['num']}, {street}, {c['area']}, {c['region']} - {c['pin']}"
    return s.upper() if rng.random() < 0.3 else s


def _fmt_phone(rng, digits):
    f = int(rng.integers(0, 3))
    return [digits, f"+91-{digits}", f"{digits[:5]} {digits[5:]}"][f]


class _Flows:
    """Accumulates invoices; each invoice has a payment lag (days) used to create payments."""
    def __init__(self):
        self.rows = []

    def add(self, issuer, receiver, amount, day, lag, item):
        self.rows.append((issuer, receiver, float(amount), int(day), float(lag), item))

    def frame(self):
        return pd.DataFrame(self.rows, columns=["issuer", "receiver", "amount", "day", "lag", "item"])


def identity_layer(rng, cfg, cid, region, reg_day, rings, groups, cowork, ring_of, group_of):
    """People, true + observed controller edges, addresses and phones.

    Shared by the pure-synthetic generator and the AML-Data loader. Only the ring / group
    structure passed in decides who shares a controller, address or phone.
    Returns (people, true_edges, rel, contacts, controller_of).
    """
    n = len(cid)
    people, true_edges = [], []

    def new_person():
        pid = f"P{len(people):05d}"
        people.append((pid, f"{rng.choice(FIRST)} {rng.choice(LAST)}", int(rng.integers(1955, 2000))))
        return pid

    controller_of = {}
    group_owner = {g["id"]: new_person() for g in groups}
    ring_ctrl = {r["id"]: new_person() for r in rings}
    for i in range(n):
        start = int(reg_day[i] + rng.integers(0, 30))
        if i in ring_of:
            r = ring_of[i]
            c = ring_ctrl[r["id"]]
            controller_of[i] = c
            t = "OWNS" if rng.random() < 0.7 else "DIRECTS"
            true_edges.append((c, cid[i], t, start))
            nominee = new_person()                       # unique nominee per member
            true_edges.append((nominee, cid[i], "DIRECTS" if t == "OWNS" else "OWNS", start))
        elif i in group_of and group_of[i].startswith("G"):
            c = group_owner[group_of[i]]
            controller_of[i] = c
            true_edges.append((c, cid[i], "OWNS", start))
            true_edges.append((c if rng.random() < 0.5 else new_person(), cid[i], "DIRECTS", start))
        else:
            c = new_person()
            controller_of[i] = c
            true_edges.append((c, cid[i], "OWNS", start))
            true_edges.append((c if rng.random() < 0.5 else new_person(), cid[i], "DIRECTS", start))
    true_edges = pd.DataFrame(true_edges, columns=["person_id", "company_id", "relationship_type", "start_day"])

    # observed = partial and noisy view of the true edges
    obs = true_edges[rng.random(len(true_edges)) < cfg.controller_obs_rate].copy()
    # true edges are 'registry' 60% of the time, 'fuzzy' 40%; confidence distributions overlap
    isreg = rng.random(len(obs)) < 0.6
    obs["source"] = np.where(isreg, "registry", "fuzzy_match")
    obs["confidence"] = np.where(isreg, rng.uniform(0.7, 1.0, len(obs)), rng.uniform(0.45, 0.9, len(obs)))
    n_noise = int(cfg.controller_noise_rate * len(true_edges) * cfg.controller_obs_rate)
    all_pids = [p[0] for p in people]
    noise = pd.DataFrame({
        "person_id": rng.choice(all_pids, n_noise),
        "company_id": rng.choice(cid, n_noise),
        "relationship_type": rng.choice(list(CONTROLLER_TYPES), n_noise),
        "start_day": -rng.integers(0, 1500, n_noise),
    })
    noise["source"] = np.where(rng.random(n_noise) < 0.7, "fuzzy_match", "self_declared")
    noise["confidence"] = rng.uniform(0.4, 0.8, n_noise)
    rel = pd.concat([obs, noise], ignore_index=True)
    rel = rel.rename(columns={"person_id": "source_id", "company_id": "target_id"})
    rel["start_date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(rel["start_day"], unit="D")
    rel["end_date"] = pd.NaT
    rel.insert(0, "rel_id", [f"REL{i:06d}" for i in range(len(rel))])
    rel = rel[["rel_id", "source_id", "target_id", "relationship_type", "start_date", "end_date",
               "source", "confidence", "start_day"]]
    people = pd.DataFrame(people, columns=["person_id", "name", "birth_year"])

    # ---------------------------------------------------------------- addresses & contacts
    canon = [_canon_address(rng, region[i]) for i in range(n)]
    phone = [f"{int(rng.integers(7, 10))}{int(rng.integers(0, 10**9)):09d}" for _ in range(n)]
    # near-duplicate benign addresses (same street, neighbouring number): fuzzy-match traps
    for i in range(n):
        if rng.random() < cfg.near_duplicate_address_p:
            j = int(rng.integers(0, n))
            c = dict(canon[j])
            c["num"] = c["num"] + int(rng.choice([-1, 1]))
            canon[i] = c
    for w in cowork:
        site = _canon_address(rng, region[int(w["members"][0])])
        site_phone = phone[int(w["members"][0])]
        for i in w["members"]:
            canon[int(i)] = dict(site)
            if rng.random() < 0.3:
                phone[int(i)] = site_phone
    for g in groups:
        m = [int(i) for i in g["members"]]
        if rng.random() < 0.5:
            for i in m[1:]:
                canon[i] = dict(canon[m[0]])
        if rng.random() < 0.3:
            for i in m[1:]:
                phone[i] = phone[m[0]]
    for r in rings:
        m = [int(i) for i in r["members"]]
        if rng.random() < cfg.ring_shared_address_p:
            sub = rng.choice(m, max(2, int(len(m) * rng.uniform(0.5, 1.0))), replace=False)
            for i in sub:
                canon[int(i)] = dict(canon[m[0]])
        if rng.random() < cfg.ring_shared_contact_p:
            sub = rng.choice(m, max(2, int(len(m) * rng.uniform(0.5, 1.0))), replace=False)
            for i in sub:
                phone[int(i)] = phone[m[0]]
    contacts = pd.DataFrame({
        "company_id": cid,
        "raw_address": [_fmt_address(rng, canon[i]) for i in range(n)],
        "raw_phone": [_fmt_phone(rng, phone[i]) for i in range(n)],
        "record_source": "registry_filing",
        "start_day": reg_day,
    })

    return people, true_edges, rel, contacts, controller_of


def generate(cfg: GenConfig) -> Dataset:
    rng = np.random.default_rng(cfg.seed)
    n, H = cfg.n_companies, cfg.horizon_days
    cid = np.array([f"C{i:05d}" for i in range(n)])

    sector = rng.choice(SECTOR_LIST, n)
    size = rng.choice(SIZES, n, p=[0.55, 0.33, 0.12])
    archetype = rng.choice(ARCHETYPES, n, p=[0.78, 0.08, 0.07, 0.07])
    region = rng.choice(REGIONS, n)
    reg_day = -rng.integers(30, 365 * 15, n)
    young = rng.random(n) < 0.10
    reg_day[young] = -rng.integers(30, 365, n)[young]

    # ---------------------------------------------------------------- group assignment
    perm = rng.permutation(n)
    cursor = 0
    rings = []                       # list of dicts
    target = int(cfg.ring_fraction * n)
    used = 0
    while used < target:
        k = int(rng.integers(cfg.ring_size[0], cfg.ring_size[1] + 1))
        members = perm[cursor:cursor + k]
        cursor += k
        used += k
        typ = str(rng.choice(TYPOLOGIES, p=np.array(cfg.typology_probs) / sum(cfg.typology_probs)))
        rings.append(dict(id=f"R{len(rings):03d}", members=members, typology=typ,
                          seed=int(rng.integers(0, 2**31 - 1))))
    groups = []
    gtarget, gused = int(cfg.benign_group_fraction * n), 0
    while gused < gtarget:
        k = int(rng.integers(2, 5))
        groups.append(dict(id=f"G{len(groups):03d}", members=perm[cursor:cursor + k]))
        cursor += k
        gused += k
    cowork = []
    for s in range(cfg.coworking_sites):
        k = int(rng.integers(cfg.coworking_size[0], cfg.coworking_size[1] + 1))
        cowork.append(dict(id=f"W{s:03d}", members=perm[cursor:cursor + k]))
        cursor += k

    ring_of = {int(i): r for r in rings for i in r["members"]}
    group_of = {int(i): g["id"] for g in groups for i in g["members"]}
    for w in cowork:
        for i in w["members"]:
            group_of[int(i)] = w["id"]
    for r in rings:                              # ring members are younger on average
        for i in r["members"]:
            if rng.random() < 0.4:
                reg_day[i] = -rng.integers(60, 500)

    people, true_edges, rel, contacts, controller_of = identity_layer(
        rng, cfg, cid, region, reg_day, rings, groups, cowork, ring_of, group_of)

    # ---------------------------------------------------------------- accounts
    acc_rows, acc_of = [], {}
    for i in range(n):
        k = 1 if rng.random() < 0.7 else 2
        acc_of[i] = []
        for _ in range(k):
            aid = f"A{len(acc_rows):06d}"
            acc_rows.append((aid, cid[i]))
            acc_of[i].append(aid)
    accounts = pd.DataFrame(acc_rows, columns=["account_id", "linked_entity_id"])

    # ---------------------------------------------------------------- benign invoices
    fl = _Flows()
    by_sector = {s: np.where(sector == s)[0] for s in SECTOR_LIST}
    seasonal_w = {}
    for i in range(n):
        if archetype[i] == "seasonal":
            ph = rng.uniform(0, 2 * np.pi)
            w = np.exp(1.8 * np.cos(2 * np.pi * np.arange(H) / 365 - ph))
            seasonal_w[i] = w / w.sum()
    for i in range(n):
        pool = np.concatenate([by_sector[s] for s in SECTOR_PARTNERS[sector[i]]])
        pool = pool[pool != i]
        mult = {"normal": 1.0, "seasonal": 0.9, "cash_heavy": 1.0, "wholesaler": 2.5}[archetype[i]]
        n_inv = int(rng.poisson(SIZE_RATE[size[i]] * mult * H / 365))
        k = 3 + int(rng.poisson(3)) + (12 if archetype[i] == "wholesaler" else 0)
        cust = rng.choice(pool, min(k, len(pool)), replace=False)
        wts = 1.0 / (np.arange(len(cust)) + 1.0)
        wts /= wts.sum()
        r = rng.choice(cust, n_inv, p=wts)
        d = rng.choice(H, n_inv, p=seasonal_w[i]) if i in seasonal_w else rng.integers(0, H, n_inv)
        amt = np.exp(rng.normal(SIZE_MU[size[i]], 0.9, n_inv))
        p_round = {"cash_heavy": 0.35, "wholesaler": 0.10}.get(archetype[i], 0.05)
        rnd = rng.random(n_inv) < p_round
        amt = np.where(rnd & (amt >= 5000), np.round(amt / 1000) * 1000, amt)
        p_mis = 0.15 if archetype[i] == "wholesaler" else 0.02
        items = [rng.choice(SECTORS[sector[i]]) if rng.random() > p_mis else rng.choice(list(ITEM_SECTOR))
                 for _ in range(n_inv)]
        lag = rng.gamma(4, 7, n_inv)
        for a, b, c, e, f, g in zip([cid[i]] * n_inv, cid[r], amt, d, lag, items):
            fl.add(a, b, c, e, f, g)
    # intra-group and reciprocal legitimate trade
    for g in groups:
        m = [int(i) for i in g["members"]]
        for _ in range(int(rng.integers(10, 30))):
            a, b = rng.choice(m, 2, replace=False)
            fl.add(cid[a], cid[b], np.exp(rng.normal(SIZE_MU[size[a]], 0.8)), rng.integers(0, H),
                   rng.gamma(4, 7), rng.choice(SECTORS[sector[a]]))
    for _ in range(int(0.03 * n)):
        a, b = rng.choice(n, 2, replace=False)
        for _ in range(int(rng.integers(4, 10))):
            for x, y in ((a, b), (b, a)):
                fl.add(cid[x], cid[y], np.exp(rng.normal(SIZE_MU[size[x]], 0.8)), rng.integers(0, H),
                       rng.gamma(4, 7), rng.choice(SECTORS[sector[x]]))

    # ---------------------------------------------------------------- ring injections
    inten = cfg.ring_intensity
    dormant_cut = {}
    other_items = list(ITEM_SECTOR)
    for r in rings:
        rr = np.random.default_rng(r["seed"])
        m = [int(i) for i in r["members"]]
        typ = r["typology"]
        if typ == "circular":
            order = list(rr.permutation(m))
            rounds = int(rr.integers(8, 15) * inten)
            for k in range(rounds):
                base_day = int(k * (H - 20) / max(rounds, 1) + rr.integers(0, 10))
                a0 = np.exp(rr.normal(12.5, 0.4))
                for h in range(len(order)):
                    x, y = order[h], order[(h + 1) % len(order)]
                    amt = a0 * (1 + rr.normal(0, 0.02))
                    if rr.random() < 0.6:
                        amt = round(amt / 1000) * 1000
                    fl.add(cid[x], cid[y], amt, min(H - 1, base_day + h * int(rr.integers(1, 3))),
                           rr.uniform(1, 5), rr.choice([it for it in other_items if ITEM_SECTOR[it] != sector[x]]))
        elif typ == "passthrough":
            hub, rest = m[0], m[1:]
            src, snk = rest[:len(rest) // 2], rest[len(rest) // 2:]
            for _ in range(int(rr.integers(12, 22) * inten)):
                s, t = int(rr.choice(src)), int(rr.choice(snk))
                a = np.exp(rr.normal(12.3, 0.4))
                if rr.random() < 0.5:
                    a = round(a / 1000) * 1000
                d = int(rr.integers(0, H - 8))
                fl.add(cid[hub], cid[s], a, d, 1.0, rr.choice(other_items))
                fl.add(cid[t], cid[hub], a * rr.uniform(0.95, 0.99), d + 1, rr.uniform(0.5, 2.0),
                       rr.choice(other_items))
        elif typ == "dormant_spike":
            d0 = int(rr.integers(150, 230))
            dormant_cut[r["id"]] = d0
            for x in m:
                for _ in range(int(rr.integers(35, 70) * inten)):
                    y = int(rr.choice(m)) if rr.random() < 0.7 else int(rr.integers(0, n))
                    if y == x:
                        continue
                    a = np.exp(rr.normal(12.2, 0.5))
                    if rr.random() < 0.5:
                        a = round(a / 1000) * 1000
                    fl.add(cid[x], cid[y], a, d0 + int(rr.integers(0, 25)), rr.uniform(1, 6),
                           rr.choice(other_items))
        else:   # camouflaged: normal-looking amounts, lags, categories; mostly structural signal
            order = list(rr.permutation(m))
            for k in range(int(rr.integers(4, 8) * inten)):
                for h in range(len(order)):
                    x, y = order[h], order[(h + 1) % len(order)]
                    fl.add(cid[x], cid[y], np.exp(rr.normal(SIZE_MU[size[x]], 0.9)),
                           int(rr.integers(0, H)), rr.gamma(4, 7), rr.choice(SECTORS[sector[x]]))

    inv = fl.frame()
    # dormant rings: strip benign activity before the spike
    idx_of = {c: i for i, c in enumerate(cid)}
    for r in rings:
        if r["typology"] == "dormant_spike":
            ids = {cid[int(i)] for i in r["members"]}
            d0 = dormant_cut[r["id"]]
            inv = inv[~((inv.issuer.isin(ids) | inv.receiver.isin(ids)) & (inv.day < d0 - 1))]
    inv = inv[inv.issuer != inv.receiver].reset_index(drop=True)
    inv.insert(0, "invoice_id", [f"INV{i:07d}" for i in range(len(inv))])
    inv["issue_date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(inv["day"], unit="D")

    # ---------------------------------------------------------------- payments
    pay_day = inv["day"].to_numpy() + inv["lag"].to_numpy()
    paid = pay_day < H
    p = inv[paid]
    sender = [acc_of[idx_of[c]][int(rng.integers(0, len(acc_of[idx_of[c]])))] for c in p.receiver]
    recv = [acc_of[idx_of[c]][int(rng.integers(0, len(acc_of[idx_of[c]])))] for c in p.issuer]
    t = pay_day[paid]
    tx = pd.DataFrame({
        "transaction_id": [f"TX{i:07d}" for i in range(len(p))],
        "sender_account": sender, "receiver_account": recv,
        "amount": p["amount"].to_numpy(), "t": t,
        "day": np.floor(t).astype(int), "invoice_id": p["invoice_id"].to_numpy(),
    })
    tx["timestamp"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(tx["t"], unit="D")
    inv = inv.drop(columns=["lag"]).rename(columns={"issuer": "issuer_id", "receiver": "receiver_id",
                                                     "item": "item_category"})

    # ---------------------------------------------------------------- tables
    companies = pd.DataFrame({
        "company_id": cid, "name": make_company_names(cfg.seed, sector), "sector": sector, "declared_size": size,
        "registration_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(reg_day, unit="D"),
        "reg_day": reg_day, "region": region, "status": "active",
    })
    gt = pd.DataFrame({"entity_id": cid, "ring_id": None, "controller_id": [controller_of[i] for i in range(n)],
                       "typology": "none", "injection_seed": -1, "synthetic_label": 0,
                       "archetype": archetype,
                       "group_id": [group_of.get(i) for i in range(n)],
                       "in_eval": True})
    for r in rings:
        for i in r["members"]:
            gt.loc[int(i), ["ring_id", "typology", "injection_seed", "synthetic_label"]] = \
                [r["id"], r["typology"], r["seed"], 1]
    meta = dict(config=cfg.to_dict(), n_rings=len(rings),
                n_ring_members=int(gt.synthetic_label.sum()),
                prevalence=float(gt.synthetic_label.mean()),
                typology_counts=pd.Series([r["typology"] for r in rings]).value_counts().to_dict(),
                synthetic=True)
    return Dataset(companies, people, accounts, rel, contacts, inv, tx, gt, true_edges, meta)
