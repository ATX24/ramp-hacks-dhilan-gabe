"""Deterministic latent worlds and sealed model-visible public views."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Literal

from distillery.contracts.hashing import content_sha256


@dataclass(frozen=True)
class Account:
    code: str
    name: str
    domain: str
    normal_balance: Literal["debit", "credit"]


@dataclass(frozen=True)
class PolicyVersion:
    policy_id: str
    version: str
    effective_from: str
    effective_to: str | None
    threshold_minor: int
    action_above: Literal["approve", "review", "reject"]
    action_at_or_below: Literal["approve", "review", "reject"]
    rule_text: str
    superseded: bool


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
    entity_id: str
    entity_name: str
    domain: str
    as_of: str
    period: str
    historical_policy_date: str
    accounts: tuple[Account, ...]
    policies: tuple[PolicyVersion, ...]
    ledger: tuple[LedgerRow, ...]
    merchants: tuple[Merchant, ...]
    book_entries: tuple[BookEntry, ...]
    bank_events: tuple[BankEvent, ...]
    variance_drivers: tuple[VarianceDriver, ...]
    missing_fields: tuple[str, ...] = ()

    def private_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "finance_agent.latent_world.v1",
            "world_id": self.world_id,
            "group_id": self.group_id,
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "domain": self.domain,
            "as_of": self.as_of,
            "period": self.period,
            "historical_policy_date": self.historical_policy_date,
            "accounts": [asdict(account) for account in self.accounts],
            "policies": [asdict(policy) for policy in self.policies],
            "ledger": [asdict(row) for row in self.ledger],
            "merchants": [asdict(merchant) for merchant in self.merchants],
            "book_entries": [asdict(entry) for entry in self.book_entries],
            "bank_events": [asdict(event) for event in self.bank_events],
            "variance_drivers": [asdict(driver) for driver in self.variance_drivers],
            "missing_fields": list(self.missing_fields),
        }

    def latent_state_hash(self) -> str:
        return f"sha256:{content_sha256(self.private_payload())}"

    def public_view(self) -> dict[str, Any]:
        """Facts and identifiers the model may use to form grounded tool arguments."""
        policy_ids = sorted({policy.policy_id for policy in self.policies})
        policy_windows = [
            {
                "policy_id": policy.policy_id,
                "version": policy.version,
                "effective_from": policy.effective_from,
                "effective_to": policy.effective_to,
            }
            for policy in sorted(
                self.policies,
                key=lambda value: (value.policy_id, value.effective_from, value.version),
            )
        ]
        return {
            "schema_version": "finance_agent.public_world.v1",
            "entity": {"entity_id": self.entity_id, "name": self.entity_name},
            "domain": self.domain,
            "as_of": self.as_of,
            "periods": [self.period],
            "historical_policy_date": self.historical_policy_date,
            "accounts": [
                {
                    "code": account.code,
                    "name": account.name,
                    "domain": account.domain,
                    "normal_balance": account.normal_balance,
                }
                for account in self.accounts
            ],
            "policy_ids": policy_ids,
            "policy_windows": policy_windows,
            "merchant_candidates": [
                {
                    "merchant_id": merchant.merchant_id,
                    "legal_name": merchant.legal_name,
                    "aliases": list(merchant.aliases),
                }
                for merchant in self.merchants
            ],
            "ledger_entry_ids": [row.entry_id for row in self.ledger],
            "reconciliation_sets": [
                {
                    "book_ids": [entry.book_id for entry in self.book_entries],
                    "bank_ids": [event.bank_id for event in self.bank_events],
                    "tolerance_minor": 0,
                }
            ],
            "variance_targets": [
                {
                    "account_code": self.variance_drivers[0].account_code,
                    "period": self.variance_drivers[0].period,
                    "top_k": len(self.variance_drivers),
                }
            ],
            "missing_fields": list(self.missing_fields),
        }


_DOMAIN_PROFILES: dict[str, dict[str, Any]] = {
    "travel": {
        "account": Account("6100", "Travel and Meals", "travel", "debit"),
        "merchant_noun": "Travel Collective",
        "memo": "team travel and meal expense",
        "policy_noun": "travel",
        "threshold_minor": 7_500,
    },
    "software": {
        "account": Account("6200", "Software Subscriptions", "software", "debit"),
        "merchant_noun": "Cloud Systems",
        "memo": "annual software subscription",
        "policy_noun": "software",
        "threshold_minor": 250_000,
    },
    "facilities": {
        "account": Account("6300", "Facilities and Rent", "facilities", "debit"),
        "merchant_noun": "Property Services",
        "memo": "monthly facilities service",
        "policy_noun": "facilities",
        "threshold_minor": 500_000,
    },
    "payroll": {
        "account": Account("6400", "Payroll Tax Expense", "payroll", "debit"),
        "merchant_noun": "Revenue Department",
        "memo": "payroll tax remittance",
        "policy_noun": "payroll tax",
        "threshold_minor": 1_000_000,
    },
}

_WORDS_LEFT = (
    "Alder",
    "Amber",
    "Birch",
    "Cobalt",
    "Coral",
    "Elm",
    "Ember",
    "Fern",
    "Flint",
    "Harbor",
    "Hazel",
    "Indigo",
    "Juniper",
    "Linden",
    "Maple",
    "Onyx",
    "Quartz",
    "River",
    "Sable",
    "Silver",
    "Spruce",
    "Stone",
    "Willow",
    "Wren",
)
_WORDS_RIGHT = (
    "Analytics",
    "Collective",
    "Commerce",
    "Consulting",
    "Exchange",
    "Foundry",
    "Goods",
    "Group",
    "Holdings",
    "Labs",
    "Logistics",
    "Networks",
    "Partners",
    "Platform",
    "Services",
    "Solutions",
    "Systems",
    "Technologies",
    "Ventures",
    "Works",
)


def _hex_token(seed: int, salt: str, nbytes: int = 9) -> str:
    return hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest()[: nbytes * 2]


def _name(seed: int, index: int, salt: str) -> str:
    digest = hashlib.sha256(f"{seed}:{index}:{salt}".encode()).digest()
    first = _WORDS_LEFT[digest[0] % len(_WORDS_LEFT)]
    second = _WORDS_LEFT[digest[1] % len(_WORDS_LEFT)]
    third = _WORDS_RIGHT[digest[2] % len(_WORDS_RIGHT)]
    return f"{first} {second} {third}"


def build_agent_world(
    *,
    seed: int,
    index: int,
    domain: str,
    ambiguous_merchant: bool = False,
    policy_conflict: bool = False,
    missing_fields: tuple[str, ...] = (),
) -> AgentWorld:
    """Build one deterministic world without filesystem, shell, or network I/O."""
    if domain not in _DOMAIN_PROFILES:
        raise ValueError(f"unknown finance domain {domain!r}")
    profile = _DOMAIN_PROFILES[domain]
    expense_account: Account = profile["account"]
    cash_account = Account("1000", "Cash", "cash", "debit")
    world_id = f"world_{_hex_token(seed, f'w{index}')}"
    group_id = f"grp_{_hex_token(seed, f'g{index}')}"
    entity_id = f"ent_{_hex_token(seed, f'e{index}')}"
    entity_name = _name(seed, index, "entity")
    month = (index % 12) + 1
    as_of = f"2026-{month:02d}-15"
    period = f"2026-{month:02d}"
    historical_policy_date = f"2025-{month:02d}-15"
    policy_id = f"pol_{domain}_threshold"
    threshold = int(profile["threshold_minor"])

    current_policy = PolicyVersion(
        policy_id=policy_id,
        version="v2",
        effective_from="2026-01-01",
        effective_to=None,
        threshold_minor=threshold,
        action_above="review",
        action_at_or_below="approve",
        rule_text=(
            f"{profile['policy_noun'].title()} expenses over {threshold} minor units "
            "require review; amounts at or below the threshold are approved."
        ),
        superseded=False,
    )
    historical_policy = PolicyVersion(
        policy_id=policy_id,
        version="v1",
        effective_from="2024-01-01",
        effective_to="2025-12-31",
        threshold_minor=max(1, threshold - max(1_000, threshold // 10)),
        action_above="reject",
        action_at_or_below="approve",
        rule_text=(
            f"Historical {profile['policy_noun']} expenses above the v1 threshold were rejected."
        ),
        superseded=True,
    )

    merchant_name = f"{_name(seed, index, 'merchant')} {profile['merchant_noun']}"
    merchant_a = Merchant(
        merchant_id=f"mrc_{_hex_token(seed, f'ma{index}')}",
        legal_name=merchant_name,
        aliases=(merchant_name.upper(), merchant_name.replace(" ", "*").upper()),
        ambiguous_group="shared_descriptor" if ambiguous_merchant else None,
    )
    merchant_b = Merchant(
        merchant_id=f"mrc_{_hex_token(seed, f'mb{index}')}",
        legal_name=f"{merchant_name} Regional",
        aliases=(merchant_name.upper(), f"{merchant_name.upper()}*REGIONAL"),
        ambiguous_group="shared_descriptor" if ambiguous_merchant else None,
    )
    merchants = (merchant_a, merchant_b) if ambiguous_merchant else (merchant_a,)

    if policy_conflict:
        amount = threshold + 1_000 + seed % 997
    else:
        amount = max(1_000, threshold // 2) + seed % 997
    if ambiguous_merchant:
        first_amount = amount
        second_amount = max(500, amount // 3)
        ledger = (
            LedgerRow(
                entry_id=f"led_{_hex_token(seed, f'la{index}')}",
                account_code=expense_account.code,
                period=period,
                merchant_id=merchant_a.merchant_id,
                amount_minor=first_amount,
                memo=str(profile["memo"]),
            ),
            LedgerRow(
                entry_id=f"led_{_hex_token(seed, f'lb{index}')}",
                account_code=expense_account.code,
                period=period,
                merchant_id=merchant_b.merchant_id,
                amount_minor=second_amount,
                memo=f"regional {profile['memo']}",
            ),
        )
    else:
        ledger = (
            LedgerRow(
                entry_id=f"led_{_hex_token(seed, f'l{index}')}",
                account_code=expense_account.code,
                period=period,
                merchant_id=merchant_a.merchant_id,
                amount_minor=amount,
                memo=str(profile["memo"]),
            ),
        )

    book_amount_a = max(1, amount // 2)
    book_amount_b = amount - book_amount_a
    book_entries = (
        BookEntry(
            book_id=f"bok_{_hex_token(seed, f'bka{index}')}",
            amount_minor=book_amount_a,
            merchant_id=merchant_a.merchant_id,
        ),
        BookEntry(
            book_id=f"bok_{_hex_token(seed, f'bkb{index}')}",
            amount_minor=book_amount_b,
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

    driver_one = -(amount // 2)
    driver_two = amount // 5
    driver_three = -(amount - abs(driver_one)) - driver_two
    variance_drivers = (
        VarianceDriver(
            driver_id=f"drv_{_hex_token(seed, f'd1{index}')}",
            account_code=expense_account.code,
            period=period,
            impact_minor=driver_one,
            label="volume",
        ),
        VarianceDriver(
            driver_id=f"drv_{_hex_token(seed, f'd2{index}')}",
            account_code=expense_account.code,
            period=period,
            impact_minor=driver_two,
            label="mix",
        ),
        VarianceDriver(
            driver_id=f"drv_{_hex_token(seed, f'd3{index}')}",
            account_code=expense_account.code,
            period=period,
            impact_minor=driver_three,
            label="price",
        ),
    )
    world = AgentWorld(
        world_id=world_id,
        group_id=group_id,
        entity_id=entity_id,
        entity_name=entity_name,
        domain=domain,
        as_of=as_of,
        period=period,
        historical_policy_date=historical_policy_date,
        accounts=(expense_account, cash_account),
        policies=(historical_policy, current_policy),
        ledger=ledger,
        merchants=merchants,
        book_entries=book_entries,
        bank_events=bank_events,
        variance_drivers=variance_drivers,
        missing_fields=missing_fields,
    )
    _assert_unique_ids(world)
    return world


def _assert_unique_ids(world: AgentWorld) -> None:
    identifiers = [
        *(row.entry_id for row in world.ledger),
        *(entry.book_id for entry in world.book_entries),
        *(event.bank_id for event in world.bank_events),
        *(merchant.merchant_id for merchant in world.merchants),
        *(driver.driver_id for driver in world.variance_drivers),
    ]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("latent world contains duplicate transaction/entity identifiers")


def agent_world_from_payload(payload: dict[str, Any]) -> AgentWorld:
    """Rehydrate a private world for deterministic post-write replay validation."""
    if payload.get("schema_version") != "finance_agent.latent_world.v1":
        raise ValueError("unsupported latent world schema")
    return AgentWorld(
        world_id=payload["world_id"],
        group_id=payload["group_id"],
        entity_id=payload["entity_id"],
        entity_name=payload["entity_name"],
        domain=payload["domain"],
        as_of=payload["as_of"],
        period=payload["period"],
        historical_policy_date=payload["historical_policy_date"],
        accounts=tuple(Account(**value) for value in payload["accounts"]),
        policies=tuple(PolicyVersion(**value) for value in payload["policies"]),
        ledger=tuple(LedgerRow(**value) for value in payload["ledger"]),
        merchants=tuple(
            Merchant(
                merchant_id=value["merchant_id"],
                legal_name=value["legal_name"],
                aliases=tuple(value["aliases"]),
                ambiguous_group=value.get("ambiguous_group"),
            )
            for value in payload["merchants"]
        ),
        book_entries=tuple(BookEntry(**value) for value in payload["book_entries"]),
        bank_events=tuple(BankEvent(**value) for value in payload["bank_events"]),
        variance_drivers=tuple(VarianceDriver(**value) for value in payload["variance_drivers"]),
        missing_fields=tuple(payload.get("missing_fields", ())),
    )


def world_public_view(world: AgentWorld) -> dict[str, Any]:
    return world.public_view()


__all__ = [
    "Account",
    "AgentWorld",
    "BankEvent",
    "BookEntry",
    "LedgerRow",
    "Merchant",
    "PolicyVersion",
    "VarianceDriver",
    "agent_world_from_payload",
    "build_agent_world",
    "world_public_view",
]
