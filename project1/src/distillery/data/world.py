"""Deterministic synthetic finance world. A seeded latent state produces every
amount and label before any natural-language rendering. Model outputs never feed
the oracle."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

GENERATOR_REVISION = "finance-world-gen-v1"

CHART_OF_ACCOUNTS = [
    ("6205", "Travel - Airfare", "expense"),
    ("6210", "Travel - Lodging", "expense"),
    ("6215", "Meals & Entertainment", "expense"),
    ("6300", "Software Subscriptions", "expense"),
    ("6310", "Cloud Infrastructure", "expense"),
    ("6400", "Office Supplies", "expense"),
    ("6500", "Professional Services", "expense"),
    ("6600", "Marketing & Advertising", "expense"),
    ("1500", "Computer Equipment (Capex)", "asset"),
    ("2100", "Accounts Payable", "liability"),
    ("2110", "Corporate Card Payable", "liability"),
]

VENDORS = [
    ("United Airlines", "UNITED 0162341", "6205"),
    ("Marriott", "MARRIOTT HTL DTWN", "6210"),
    ("DoorDash", "DD *DOORDASH", "6215"),
    ("Figma", "FIGMA INC", "6300"),
    ("AWS", "AMZN WEB SERVICES", "6310"),
    ("Staples", "STAPLES #0424", "6400"),
    ("Deloitte", "DELOITTE LLP SVCS", "6500"),
    ("Google Ads", "GOOGLE *ADS4823", "6600"),
    ("Apple", "APPLE STORE R042", "1500"),
    ("Datadog", "DATADOG INC", "6310"),
    ("Uber", "UBER *TRIP", "6205"),
    ("Notion", "NOTION LABS", "6300"),
]

POLICIES = [
    ("POL-TRAVEL-017", "Airfare and ground transport under $150000 minor units auto-approve; above requires review.", "6205", 150000),
    ("POL-LODGE-004", "Lodging under $80000 minor units per night auto-approves; above requires review.", "6210", 80000),
    ("POL-MEALS-009", "Meals under $15000 minor units approve; $15000-$50000 review; above reject.", "6215", 15000),
    ("POL-SAAS-002", "Software subscriptions under $100000 minor units approve if vendor is on the approved list.", "6300", 100000),
    ("POL-CAPEX-001", "Hardware purchases at or above $250000 minor units must be capitalized to 1500 and reviewed.", "1500", 250000),
    ("POL-MKTG-011", "Marketing spend under $200000 minor units approves; above requires review.", "6600", 200000),
]

DRIVERS = [
    ("cloud_usage", "Cloud infrastructure usage", "expense"),
    ("support_volume", "Support ticket volume", "expense"),
    ("headcount", "Headcount growth", "expense"),
    ("ad_spend", "Advertising spend", "expense"),
    ("license_sales", "License sales", "revenue"),
    ("services_rev", "Services revenue", "revenue"),
    ("fx_effect", "FX remeasurement", "expense"),
]

DEPARTMENTS = ["Engineering", "Sales", "G&A", "Marketing", "Support"]


@dataclass
class Txn:
    txn_id: str
    vendor: str
    descriptor: str
    amount_minor: int
    currency: str
    date: str
    department: str
    gl_account: str
    policy_id: str
    policy_action: str  # approve | review | reject


@dataclass
class World:
    world_id: str
    group_id: str
    seed: int
    template_family: str
    txns: list[Txn] = field(default_factory=list)
    # variance: (driver_id, impact_minor) list plus other bucket
    variance_drivers: list[tuple[str, int]] = field(default_factory=list)
    variance_other_minor: int = 0
    # reconciliation latent events
    recon: dict = field(default_factory=dict)


def _policy_action(gl: str, amount: int, rng: random.Random) -> tuple[str, str]:
    for pid, _, pgl, threshold in POLICIES:
        if pgl == gl:
            if gl == "6215":
                if amount < threshold:
                    return "approve", pid
                if amount < 50000:
                    return "review", pid
                return "reject", pid
            if gl == "1500":
                return ("review", pid) if amount >= threshold else ("approve", pid)
            return ("approve", pid) if amount < threshold else ("review", pid)
    return "approve", "POL-DEFAULT-000"


def build_world(world_idx: int, regime: str = "iid") -> World:
    """Deterministic per-index world. `regime` shifts templates/policy combos for OOD."""
    seed = 100_003 * world_idx + (7 if regime == "ood" else 0)
    rng = random.Random(seed)
    wid = f"world_{regime}_{world_idx:05d}"
    w = World(world_id=wid, group_id=f"grp_{regime}_{world_idx:05d}", seed=seed,
              template_family=f"txn_policy_{'v9' if regime == 'ood' else 'v3'}")

    n_txn = rng.randint(4, 8)
    vendor_pool = VENDORS if regime == "iid" else VENDORS[::-1]
    for i in range(n_txn):
        vendor, desc, gl = vendor_pool[rng.randrange(len(vendor_pool))]
        base = rng.randint(1500, 400000)
        # hard negatives: push amounts near policy thresholds sometimes
        if rng.random() < 0.35:
            for pid, _, pgl, threshold in POLICIES:
                if pgl == gl:
                    base = threshold + rng.choice([-1, 0, 1]) * rng.randint(0, 500)
                    base = max(base, 100)
        action, pid = _policy_action(gl, base, rng)
        w.txns.append(Txn(
            txn_id=f"txn_{wid}_{i}", vendor=vendor, descriptor=desc,
            amount_minor=base, currency="USD",
            date=f"2026-0{rng.randint(1, 6)}-{rng.randint(10, 28)}",
            department=rng.choice(DEPARTMENTS), gl_account=gl,
            policy_id=pid, policy_action=action,
        ))

    n_drv = rng.randint(2, 4)
    chosen = rng.sample(DRIVERS, n_drv)
    for drv_id, _, kind in chosen:
        mag = rng.randint(20000, 500000)
        sign = -1 if (kind == "expense") == (rng.random() < 0.75) else 1
        w.variance_drivers.append((drv_id, sign * mag))
    w.variance_other_minor = rng.randint(-40000, 40000)

    # cash reconciliation latent state
    fee = rng.randint(500, 5000)
    w.recon = {
        "n_matched": rng.randint(2, 4),
        "bank_fee_minor": fee,
        "has_duplicate": rng.random() < 0.3,
        "deposit_in_transit_minor": rng.randint(10000, 90000) if rng.random() < 0.5 else 0,
        "book_balance_minor": rng.randint(5_000_000, 9_000_000),
    }
    return w
