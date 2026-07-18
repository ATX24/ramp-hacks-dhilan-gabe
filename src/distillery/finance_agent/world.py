"""Deterministic latent world for Finance Agent sandbox tools."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from distillery.contracts.hashing import content_sha256


@dataclass(frozen=True)
class Account:
    code: str
    name: str
    domain: str


@dataclass(frozen=True)
class PolicyVersion:
    policy_id: str
    version: str
    effective_from: str
    effective_to: str | None
    rule_text: str
    action: Literal["approve", "review", "reject"]
    superseded: bool = False


@dataclass(frozen=True)
class LedgerRow:
    entry_id: str
    account_code: str
    period: str
    merchant_id: str
    amount_minor: int
    memo: str


@dataclass(frozen=True)
class Merchant:
    merchant_id: str
    legal_name: str
    aliases: tuple[str, ...]
    ambiguous_group: str | None = None


@dataclass(frozen=True)
class BankEvent:
    bank_id: str
    amount_minor: int
    descriptor: str


@dataclass(frozen=True)
class BookEntry:
    book_id: str
    amount_minor: int
    merchant_id: str


@dataclass(frozen=True)
class VarianceDriver:
    driver_id: str
    account_code: str
    period: str
    impact_minor: int
    label: str


@dataclass(frozen=True)
class AgentWorld:
    world_id: str
    group_id: str
    domain: str
    as_of: str
    accounts: tuple[Account, ...]
    policies: tuple[PolicyVersion, ...]
    ledger: tuple[LedgerRow, ...]
    merchants: tuple[Merchant, ...]
    book_entries: tuple[BookEntry, ...]
    bank_events: tuple[BankEvent, ...]
    variance_drivers: tuple[VarianceDriver, ...]
    missing_fields: tuple[str, ...] = ()

    def latent_state_hash(self) -> str:
        payload = {
            "world_id": self.world_id,
            "group_id": self.group_id,
            "domain": self.domain,
            "as_of": self.as_of,
            "accounts": [account.__dict__ for account in self.accounts],
            "policies": [policy.__dict__ for policy in self.policies],
            "ledger": [row.__dict__ for row in self.ledger],
            "merchants": [merchant.__dict__ for merchant in self.merchants],
            "book_entries": [entry.__dict__ for entry in self.book_entries],
            "bank_events": [event.__dict__ for event in self.bank_events],
            "variance_drivers": [driver.__dict__ for driver in self.variance_drivers],
            "missing_fields": list(self.missing_fields),
        }
        return f"sha256:{content_sha256(payload)}"


_DOMAINS = ("travel", "software", "facilities", "payroll")


def _hex_token(seed: int, salt: str, nbytes: int = 9) -> str:
    digest = hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()
    return digest[: nbytes * 2]


def build_agent_world(
    *,
    seed: int,
    index: int,
    domain: str | None = None,
    ambiguous_merchant: bool = False,
    stale_policy: bool = False,
    missing_fields: tuple[str, ...] = (),
) -> AgentWorld:
    """Build a deterministic latent world. No I/O."""
    chosen_domain = domain or _DOMAINS[seed % len(_DOMAINS)]
    world_id = f"world_{_hex_token(seed, f'w{index}')}"
    group_id = f"grp_{_hex_token(seed, f'g{index}')}"
    month = (index % 12) + 1
    as_of = f"2026-{month:02d}-15"
    period = f"2026-{month:02d}"

    accounts = (
        Account("6100", "Travel Meals", "travel"),
        Account("6200", "Software Subscriptions", "software"),
        Account("6300", "Facilities Rent", "facilities"),
        Account("6400", "Payroll Taxes", "payroll"),
        Account("1000", "Cash", "cash"),
    )
    current_policy = PolicyVersion(
        policy_id="pol_meal_limit",
        version="v2",
        effective_from="2026-01-01",
        effective_to=None,
        rule_text="Meal expenses over 7500 minor units require review.",
        action="review",
        superseded=False,
    )
    stale = PolicyVersion(
        policy_id="pol_meal_limit",
        version="v1",
        effective_from="2024-01-01",
        effective_to="2025-12-31",
        rule_text="Meal expenses over 5000 minor units require reject.",
        action="reject",
        superseded=True,
    )
    policies = (current_policy, stale) if stale_policy else (current_policy,)

    merchant_a = Merchant(
        merchant_id=f"mrc_{_hex_token(seed, f'ma{index}')}",
        legal_name="Harbor Travel Collective",
        aliases=("HARBOR TRAVEL", "HRBR*TRAVEL"),
        ambiguous_group="harbor_travel" if ambiguous_merchant else None,
    )
    merchant_b = Merchant(
        merchant_id=f"mrc_{_hex_token(seed, f'mb{index}')}",
        legal_name="Harbor Travel Collective NY",
        aliases=("HARBOR TRAVEL NY", "HRBR*TRAVEL*NY"),
        ambiguous_group="harbor_travel" if ambiguous_merchant else None,
    )
    merchants = (merchant_a, merchant_b) if ambiguous_merchant else (merchant_a,)

    amount = 5_000 + (seed % 50) * 100 + index
    ledger = (
        LedgerRow(
            entry_id=f"led_{_hex_token(seed, f'l{index}')}",
            account_code="6100" if chosen_domain == "travel" else "6200",
            period=period,
            merchant_id=merchant_a.merchant_id,
            amount_minor=amount,
            memo="team offsite meal" if chosen_domain == "travel" else "saas seat expansion",
        ),
    )
    book_entries = (
        BookEntry(
            book_id=f"bok_{_hex_token(seed, f'bk{index}')}",
            amount_minor=amount,
            merchant_id=merchant_a.merchant_id,
        ),
    )
    bank_events = (
        BankEvent(
            bank_id=f"bnk_{_hex_token(seed, f'bn{index}')}",
            amount_minor=amount,
            descriptor=merchant_a.aliases[0],
        ),
    )
    drivers = (
        VarianceDriver(
            driver_id=f"drv_{_hex_token(seed, f'd1{index}')}",
            account_code=ledger[0].account_code,
            period=period,
            impact_minor=-(amount // 2),
            label="volume",
        ),
        VarianceDriver(
            driver_id=f"drv_{_hex_token(seed, f'd2{index}')}",
            account_code=ledger[0].account_code,
            period=period,
            impact_minor=-(amount - amount // 2),
            label="price",
        ),
    )
    return AgentWorld(
        world_id=world_id,
        group_id=group_id,
        domain=chosen_domain,
        as_of=as_of,
        accounts=accounts,
        policies=policies,
        ledger=ledger,
        merchants=merchants,
        book_entries=book_entries,
        bank_events=bank_events,
        variance_drivers=drivers,
        missing_fields=missing_fields,
    )


def world_public_view(world: AgentWorld) -> dict[str, Any]:
    """Non-latent hints safe to expose in episode input (still synthetic)."""
    return {
        "as_of": world.as_of,
        "domain": world.domain,
        "merchant_hints": [m.legal_name for m in world.merchants],
        "known_account_codes": [a.code for a in world.accounts],
        "missing_fields": list(world.missing_fields),
    }
