"""Synthetic noisy card-transaction descriptors for the demo dataset."""
from __future__ import annotations

import random

_TEMPLATES = [
    "AWS EMEA {n} LUXEMBOURG", "AMZN WEB SVCS {n}", "GITHUB INC* {n}",
    "UBER *TRIP {n} HELP.UBER.C", "UBER* EATS {n}", "DOORDASH*{n} SF CA",
    "FIGMA MONTHLY {n}", "DELTA AIR 006{n}ATL", "OPENAI *CHATGPT SUBSCR",
    "WEWORK {n} NYC", "GOOGLE ADS{n}", "STRIPE {n} PAYOUT FEE",
    "SQ *COFFEE BAR {n}", "TST* TACO SPOT {n} AUSTIN",
]


def descriptors(n: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    return [rng.choice(_TEMPLATES).format(n=rng.randint(1000, 99999)) for _ in range(n)]
