"""Writes a MOCK file in the AML-Data CSV/patterns format (as we believe it to be).
It tests the loader plumbing only - it is NOT IBM data and has none of its statistics."""
import numpy as np
import pandas as pd

HEADER = ("Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
          "Amount Paid,Payment Currency,Payment Format,Is Laundering")


def make_mock(out_dir, n_acc=15000, n_tx=120000, n_attempts=60, seed=0):
    rng = np.random.default_rng(seed)
    acc = [(f"{int(rng.integers(1, 40)):03d}", f"{int(rng.integers(0, 16**7)):07X}0") for _ in range(n_acc)]
    w = 1.0 / (np.arange(n_acc) + 10.0)
    w /= w.sum()
    t0 = pd.Timestamp("2022-09-01")
    rows = []

    def add(ts, a, b, amt, fmt, lab):
        rows.append((ts.strftime("%Y/%m/%d %H:%M"), a[0], a[1], b[0], b[1], f"{amt:.2f}", "US Dollar",
                     f"{amt:.2f}", "US Dollar", fmt, lab))

    s = rng.choice(n_acc, n_tx, p=w)
    d = rng.choice(n_acc, n_tx, p=w)
    for i in range(n_tx):
        if s[i] != d[i]:
            add(t0 + pd.Timedelta(minutes=int(rng.integers(0, 14400))), acc[s[i]], acc[d[i]],
                float(np.exp(rng.normal(8, 1.5))), str(rng.choice(["Cheque", "ACH", "Credit Card", "Wire"])), 0)
    blocks, fresh = [], 0
    for k in range(n_attempts):
        typ = str(rng.choice(["CYCLE", "FAN-OUT", "FAN-IN", "SCATTER-GATHER"]))
        m = int(rng.integers(4, 9))
        mem = [(f"{int(rng.integers(1, 40)):03d}", f"BAD{fresh + j:05X}0") for j in range(m)]
        fresh += m
        base = float(np.exp(rng.normal(11, 0.5)))
        lines = []
        ts = t0 + pd.Timedelta(minutes=int(rng.integers(0, 12000)))
        if typ == "CYCLE":
            pairs = [(mem[j], mem[(j + 1) % m]) for j in range(m)]
        elif typ == "FAN-OUT":
            pairs = [(mem[0], mem[j]) for j in range(1, m)]
        elif typ == "FAN-IN":
            pairs = [(mem[j], mem[0]) for j in range(1, m)]
        else:
            pairs = [(mem[0], mem[j]) for j in range(1, m - 1)] + [(mem[j], mem[m - 1]) for j in range(1, m - 1)]
        for a, b in pairs:
            ts = ts + pd.Timedelta(minutes=int(rng.integers(5, 600)))
            amt = base * rng.uniform(0.9, 1.0)
            add(ts, a, b, amt, "ACH", 1)
            lines.append(f"{ts.strftime('%Y/%m/%d %H:%M')},{a[0]},{a[1]},{b[0]},{b[1]},{amt:.2f},US Dollar,{amt:.2f},US Dollar,ACH,1")
        blocks.append(f"BEGIN LAUNDERING ATTEMPT - {typ}:  Max {m} hops\n" + "\n".join(lines) + f"\nEND LAUNDERING ATTEMPT - {typ}:  Max {m} hops\n")
        for a in mem:       # laundering accounts also do ordinary activity
            for _ in range(int(rng.integers(2, 8))):
                b = acc[int(rng.choice(n_acc, p=w))]
                add(t0 + pd.Timedelta(minutes=int(rng.integers(0, 14400))), a, b, float(np.exp(rng.normal(8, 1.5))), "Cheque", 0)
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed)
    with open(f"{out_dir}/Mock_Trans.csv", "w", encoding="utf-8", newline="") as f:
        f.write(HEADER + "\n")
        df.to_csv(f, header=False, index=False)
    (open(f"{out_dir}/Mock_Patterns.txt", "w", encoding="utf-8")).write("\n".join(blocks))
    return f"{out_dir}/Mock_Trans.csv", f"{out_dir}/Mock_Patterns.txt"
