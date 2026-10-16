"""Shared constants and the generator configuration."""
from dataclasses import dataclass, field, asdict

SECTORS = {
    "textiles": ["fabric", "yarn", "garments", "dyes"],
    "electronics": ["components", "pcb", "cables", "devices"],
    "food_processing": ["grain", "packaged_food", "edible_oil", "spices"],
    "chemicals": ["solvents", "pigments", "fertiliser", "plastics_resin"],
    "construction": ["cement", "steel", "tiles", "contracting"],
    "it_services": ["software_dev", "hosting", "consulting", "support"],
    "logistics": ["freight", "warehousing", "courier", "cold_chain"],
    "pharma": ["apis", "formulations", "packaging", "lab_supplies"],
}
SECTOR_LIST = list(SECTORS)
ITEM_SECTOR = {item: s for s, items in SECTORS.items() for item in items}
# Each sector sells to itself and the next two sectors in this list (a crude supply chain).
SECTOR_PARTNERS = {
    s: [SECTOR_LIST[(i + k) % len(SECTOR_LIST)] for k in range(3)]
    for i, s in enumerate(SECTOR_LIST)
}

SIZES = ["small", "medium", "large"]
SIZE_MU = {"small": 10.0, "medium": 11.0, "large": 12.0}      # log-mean invoice amount (INR)
SIZE_RATE = {"small": 22, "medium": 50, "large": 90}          # invoices issued per year
REGIONS = ["Delhi", "Maharashtra", "Karnataka", "Gujarat", "Tamil Nadu", "Uttar Pradesh"]
TYPOLOGIES = ["circular", "passthrough", "dormant_spike", "camouflaged"]
ARCHETYPES = ["normal", "seasonal", "cash_heavy", "wholesaler"]

CONTROLLER_TYPES = ("OWNS", "DIRECTS")

# Queue thresholds on the fused (calibrated) score. These are prioritisation cut-offs,
# not probabilities of wrongdoing.
HIGH_P = 0.50
REVIEW_P = 0.20


@dataclass
class GenConfig:
    seed: int = 42
    n_companies: int = 4000
    horizon_days: int = 365
    ring_fraction: float = 0.025        # share of companies that sit in injected rings
    ring_size: tuple = (4, 8)
    typology_probs: tuple = (0.3, 0.25, 0.2, 0.25)   # aligned with TYPOLOGIES
    controller_obs_rate: float = 0.5    # P(a true OWNS/DIRECTS edge is observed)
    controller_noise_rate: float = 0.15 # wrong controller edges, as a share of true edges
    benign_group_fraction: float = 0.05 # legitimate multi-company owners (hard negatives)
    coworking_sites: int = 8
    coworking_size: tuple = (15, 35)
    ring_shared_address_p: float = 0.5
    ring_shared_contact_p: float = 0.4
    near_duplicate_address_p: float = 0.03   # benign near-identical addresses (fuzzy-match traps)
    ring_intensity: float = 1.0

    def to_dict(self):
        return asdict(self)
