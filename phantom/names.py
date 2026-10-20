"""Readable FICTIONAL company names. A pure function of (seed, sector): carries no label information.
Names are random combinations and may coincide with real businesses by chance; the dashboard says so."""
import numpy as np

SURNAMES = ["Sharma", "Verma", "Gupta", "Singh", "Kumar", "Mehta", "Iyer", "Reddy", "Patel", "Nair", "Jain",
            "Agarwal", "Bose", "Khanna", "Malhotra", "Chopra", "Joshi", "Rao", "Das", "Shah", "Kapoor", "Bhatia",
            "Menon", "Saxena", "Trivedi", "Pillai", "Banerjee", "Chauhan", "Desai", "Mishra", "Kulkarni", "Sethi",
            "Arora", "Bajaj", "Ghosh", "Hegde", "Lal", "Naidu", "Oberoi", "Rastogi"]
SECTOR_WORDS = {
    "textiles": ["Textiles", "Fabrics", "Weaves", "Garments", "Mills"],
    "electronics": ["Electronics", "Circuits", "Components", "Devices", "Systems"],
    "food_processing": ["Foods", "Agro Products", "Grains", "Spices", "Edibles"],
    "chemicals": ["Chemicals", "Polymers", "Pigments", "Solvents", "Agrochem"],
    "construction": ["Constructions", "Builders", "Steel Works", "Infra", "Cement Traders"],
    "it_services": ["Infotech", "Software", "Digital Services", "Tech Solutions", "Cloudworks"],
    "logistics": ["Logistics", "Freight Lines", "Carriers", "Cold Chain", "Transport Co"],
    "pharma": ["Pharma", "Lifesciences", "Healthcare", "Formulations", "Labs"],
}
SUFFIXES = ["Pvt Ltd", "Pvt Ltd", "Pvt Ltd", "& Sons", "Enterprises", "Traders", "Industries", "LLP", "Ltd"]


def make_company_names(seed: int, sectors) -> list:
    rng = np.random.default_rng(int(seed) + 1_000_003)
    seen, out = {}, []
    for s in sectors:
        base = f"{rng.choice(SURNAMES)} {rng.choice(SECTOR_WORDS[s])} {rng.choice(SUFFIXES)}"
        k = seen.get(base, 0)
        seen[base] = k + 1
        out.append(base if k == 0 else f"{base} ({k + 1})")
    return out
