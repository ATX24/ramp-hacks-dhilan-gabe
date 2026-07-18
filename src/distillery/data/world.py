"""Deterministic latent finance world state and policy semantics."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal

from distillery.contracts.hashing import content_sha256
from distillery.contracts.tasks import Difficulty, TaskId

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NAME_LEFT = (
    "Alder",
    "Amber",
    "Birch",
    "Cedar",
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
    "Marble",
    "Navy",
    "Oak",
    "Olive",
    "Onyx",
    "Pearl",
    "Pine",
    "Quartz",
    "Reed",
    "River",
    "Sable",
    "Silver",
    "Spruce",
    "Stone",
    "Willow",
    "Wren",
)
_NAME_RIGHT = (
    "Analytics",
    "Collective",
    "Commerce",
    "Company",
    "Consulting",
    "Digital",
    "Exchange",
    "Foundry",
    "Goods",
    "Group",
    "Holdings",
    "Labs",
    "Logistics",
    "Market",
    "Networks",
    "Partners",
    "Platform",
    "Services",
    "Solutions",
    "Systems",
    "Technologies",
    "Travel",
    "Ventures",
    "Works",
)


class PolicyAction(StrEnum):
    APPROVE = "approve"
    REVIEW = "review"
    REJECT = "reject"


class VarianceRegime(StrEnum):
    SIMPLE = "simple"
    OFFSET = "offset"
    PRICE_VOLUME = "price_volume"
    FX = "fx"
    TIE = "tie"
    HIDDEN_SUBTOTAL = "hidden_subtotal"


class CashRegime(StrEnum):
    CLEAN_MATCH = "clean_match"
    BANK_FEE = "bank_fee"
    DEPOSIT_IN_TRANSIT = "deposit_in_transit"
    STALE_CHECK = "stale_check"
    DUPLICATE = "duplicate"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    PARTIAL = "partial"
    SAME_AMOUNT_COLLISION = "same_amount_collision"


class TxnHardNegative(StrEnum):
    NONE = "none"
    NEAR_SYNONYM_GL = "near_synonym_gl"
    REFUND = "refund"
    CHARGEBACK = "chargeback"
    CAPEX_OPEX = "capex_opex"
    SPLIT_ALLOCATION = "split_allocation"
    REFUND_SPLIT = "refund_split"
    PERSONAL_LOOKING_ALLOWED = "personal_looking_allowed"
    ALLOWED_LOOKING_PROHIBITED = "allowed_looking_prohibited"
    CONFLICTING_RULES = "conflicting_rules"
    THRESHOLD_BOUNDARY = "threshold_boundary"
    MISLEADING_DESCRIPTOR = "misleading_descriptor"


@dataclass(frozen=True)
class GLAccount:
    code: str
    name: str
    account_type: Literal["expense", "asset", "liability", "revenue"]


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    semantic_code: str
    precedence: int
    action: PolicyAction
    gl_account: str
    min_amount_minor: int
    max_amount_minor: int | None
    keywords: tuple[str, ...]
    category: str
    vendor_ids: tuple[str, ...]
    text: str

    def amount_applies(self, amount_minor: int) -> bool:
        amount = abs(amount_minor)
        if amount < self.min_amount_minor:
            return False
        return self.max_amount_minor is None or amount <= self.max_amount_minor


@dataclass(frozen=True)
class VendorArchetype:
    key: str
    category: str
    default_gl: str
    descriptor_term: str


@dataclass(frozen=True)
class Vendor:
    vendor_id: str
    archetype: str
    name: str
    descriptors: tuple[str, ...]
    default_gl: str
    category: str
    corruption_family: str


@dataclass(frozen=True)
class TransactionLatent:
    txn_id: str
    vendor_id: str
    descriptor: str
    amount_minor: int
    currency: str
    date: str
    entity_id: str
    cost_center: str
    transaction_kind: Literal["charge", "refund", "chargeback"]
    hard_negative: TxnHardNegative
    gl_account: str
    policy_action: PolicyAction
    applied_rule_ids: tuple[str, ...]
    contra_account: str = "2100"
    split_lines: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class VarianceObservation:
    source_id: str
    driver_id: str
    pnl_type: Literal["expense", "revenue"]
    budget_minor: int
    actual_minor: int
    kind: Literal["expense", "revenue", "fx", "volume", "price", "other"]

    def profit_impact_minor(self) -> int:
        if self.pnl_type == "expense":
            return self.budget_minor - self.actual_minor
        return self.actual_minor - self.budget_minor


@dataclass(frozen=True)
class VarianceLatent:
    period: str
    budget_minor: int
    actual_minor: int
    drivers: tuple[VarianceObservation, ...]
    unallocated: tuple[VarianceObservation, ...]
    rule_ids: tuple[str, ...]
    analysis_rules: tuple[tuple[str, str], ...]
    regime: VarianceRegime
    entity_id: str


@dataclass(frozen=True)
class BookEntry:
    entry_id: str
    amount_minor: int
    date: str
    memo: str


@dataclass(frozen=True)
class BankEvent:
    event_id: str
    amount_minor: int
    date: str
    memo: str
    event_type: Literal["clearing", "fee", "duplicate", "other"] = "clearing"


@dataclass(frozen=True)
class CashLatent:
    book_balance_minor: int
    bank_balance_minor: int
    book_entries: tuple[BookEntry, ...]
    bank_events: tuple[BankEvent, ...]
    matched: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    exceptions: tuple[tuple[str, tuple[str, ...], int], ...]
    regime: CashRegime
    entity_id: str
    close_period: str


@dataclass(frozen=True)
class LatentWorld:
    """Internally consistent company snapshot for one synthetic world."""

    world_id: str
    group_id: str
    entity_id: str
    close_period: str
    chart_of_accounts: tuple[GLAccount, ...]
    vendors: tuple[Vendor, ...]
    policies: tuple[PolicyRule, ...]
    departments: tuple[str, ...]
    transaction: TransactionLatent | None = None
    variance: VarianceLatent | None = None
    cash: CashLatent | None = None
    latent_regime: str = "baseline"
    ood_held_out: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def latent_payload(self) -> dict[str, Any]:
        return asdict(self)

    def latent_state_hash(self) -> str:
        return f"sha256:{content_sha256(self.latent_payload())}"


CHART_ACCOUNT_SPECS: tuple[
    tuple[str, str, Literal["expense", "asset", "liability", "revenue"]], ...
] = (
    ("1500", "Property and equipment", "asset"),
    ("2100", "Accounts payable", "liability"),
    ("4000", "Operating revenue", "revenue"),
    ("6100", "Meals and entertainment", "expense"),
    ("6105", "Office refreshments", "expense"),
    ("6205", "Air travel", "expense"),
    ("6210", "Lodging", "expense"),
    ("6400", "Software subscriptions", "expense"),
    ("6500", "Cloud infrastructure", "expense"),
    ("6600", "Professional services", "expense"),
    ("6700", "Bank and processor fees", "expense"),
)

IID_VENDOR_ARCHETYPES: tuple[VendorArchetype, ...] = (
    VendorArchetype("cafe", "meals", "6100", "cafe"),
    VendorArchetype("airline", "airfare", "6205", "air"),
    VendorArchetype("cloud", "cloud", "6500", "compute"),
    VendorArchetype("hardware", "capex", "1500", "equipment"),
    VendorArchetype("saas", "saas", "6400", "subscription"),
    VendorArchetype("rideshare", "meals", "6100", "offsite"),
    VendorArchetype("hotel", "lodging", "6210", "hotel"),
    VendorArchetype("consulting", "services", "6600", "consulting"),
    VendorArchetype("personal", "personal", "6100", "marketplace"),
)

OOD_VENDOR_ARCHETYPES: tuple[VendorArchetype, ...] = (
    VendorArchetype("workspace_alt", "facilities", "6400", "workspace"),
    VendorArchetype("processor_alt", "fees", "6700", "processing"),
    VendorArchetype("cloud_alt", "cloud", "6500", "capacity"),
    VendorArchetype("hotel_alt", "lodging", "6210", "inn"),
    VendorArchetype("equipment_alt", "capex", "1500", "machinery"),
    VendorArchetype("consulting_alt", "services", "6600", "advisory"),
    VendorArchetype("cafe_alt", "meals", "6100", "catering"),
    VendorArchetype("personal_alt", "personal", "6100", "consumer"),
)

# Compatibility aliases used by downstream data-only callers.
IID_VENDORS = IID_VENDOR_ARCHETYPES
OOD_VENDORS = OOD_VENDOR_ARCHETYPES

IID_TEMPLATE_FAMILIES: dict[TaskId, tuple[str, ...]] = {
    TaskId.TRANSACTION_REVIEW: (
        "txn_card_packet",
        "txn_policy_memo",
        "txn_ledger_excerpt",
        "txn_receipt_bundle",
    ),
    TaskId.VARIANCE_ANALYSIS: (
        "var_operating_table",
        "var_driver_packet",
        "var_close_memo",
        "var_dimension_slice",
    ),
    TaskId.CASH_RECONCILIATION: (
        "cash_statement_packet",
        "cash_close_table",
        "cash_ledger_extract",
    ),
}

OOD_TEMPLATE_FAMILIES: dict[TaskId, tuple[str, ...]] = {
    TaskId.TRANSACTION_REVIEW: (
        "txn_matrix_packet",
        "txn_account_crosswalk",
        "txn_control_note",
    ),
    TaskId.VARIANCE_ANALYSIS: (
        "var_bridge_packet",
        "var_unit_economics",
        "var_currency_schedule",
    ),
    TaskId.CASH_RECONCILIATION: (
        "cash_aggregation_sheet",
        "cash_collision_worksheet",
        "cash_settlement_packet",
    ),
}


def normalize_policy_tokens(text: str) -> tuple[str, ...]:
    """Normalize policy matching to complete alphanumeric tokens."""
    return tuple(_TOKEN_RE.findall(text.casefold()))


def phrase_matches(tokens: Sequence[str], phrase: str) -> bool:
    """Match a normalized complete token or contiguous token phrase."""
    wanted = normalize_policy_tokens(phrase)
    if not wanted or len(wanted) > len(tokens):
        return False
    width = len(wanted)
    return any(tuple(tokens[i : i + width]) == wanted for i in range(len(tokens) - width + 1))


def policy_match_rank(
    rule: PolicyRule,
    *,
    vendor_id: str,
    category: str,
    descriptor: str,
) -> int | None:
    """Vendor match wins, then category, then complete token/phrase match."""
    if vendor_id and vendor_id in rule.vendor_ids:
        return 0
    if category and category == rule.category:
        return 1
    tokens = normalize_policy_tokens(descriptor)
    if any(phrase_matches(tokens, phrase) for phrase in rule.keywords):
        return 2
    return None


def applicable_policy_rules(
    policies: Sequence[PolicyRule],
    *,
    descriptor: str,
    amount_minor: int,
    category: str,
    vendor_id: str = "",
) -> tuple[PolicyRule, ...]:
    ranked: list[tuple[int, int, str, PolicyRule]] = []
    for rule in policies:
        rank = policy_match_rank(
            rule,
            vendor_id=vendor_id,
            category=category,
            descriptor=descriptor,
        )
        if rank is None or not rule.amount_applies(amount_minor):
            continue
        ranked.append((rank, rule.precedence, rule.rule_id, rule))
    ranked.sort(key=lambda item: item[:3])
    return tuple(item[3] for item in ranked)


def resolve_policy(
    policies: Sequence[PolicyRule],
    *,
    descriptor: str,
    amount_minor: int,
    category: str,
    vendor_id: str = "",
    hard_negative: TxnHardNegative | None = None,
) -> tuple[PolicyAction, str, tuple[str, ...]]:
    """Resolve only applicable rules; hard-negative labels never affect policy logic."""
    del hard_negative
    matches = applicable_policy_rules(
        policies,
        descriptor=descriptor,
        amount_minor=amount_minor,
        category=category,
        vendor_id=vendor_id,
    )
    if not matches:
        raise ValueError(
            "no applicable policy rule for "
            f"vendor={vendor_id!r} category={category!r} amount={amount_minor}"
        )
    winner = matches[0]
    return winner.action, winner.gl_account, (winner.rule_id,)


def build_world(
    *,
    seed: int,
    index: int,
    split_token: str,
    task: TaskId,
    difficulty: Difficulty,
    ood: bool = False,
    salt: int = 0,
) -> LatentWorld:
    """Build one deterministic world; split domains are hashed out of visible IDs."""
    identity = f"{seed}|{split_token}|{index}|{salt}|{task.value}|{difficulty.value}|{int(ood)}"
    group_identity = f"{seed}|{split_token}|{index // 4}|{salt}|group"
    h = _hash_int(identity)
    world_id = _opaque_id("world_", identity)
    group_id = _opaque_id("grp_", group_identity)
    entity_id = _opaque_id("ent_", f"{group_identity}|entity")
    close_period = _period_label(identity)
    chart = _build_chart(group_identity, ood=ood)
    vendors = _build_vendors(group_identity, ood=ood)
    policies = _build_policies(group_identity, vendors, ood=ood)

    transaction: TransactionLatent | None = None
    variance: VarianceLatent | None = None
    cash: CashLatent | None = None
    latent_regime = "baseline"

    if task == TaskId.TRANSACTION_REVIEW:
        transaction = _build_transaction(
            h,
            identity,
            vendors,
            policies,
            entity_id,
            difficulty,
        )
        latent_regime = transaction.hard_negative.value
    elif task == TaskId.VARIANCE_ANALYSIS:
        variance = _build_variance(
            h,
            identity,
            entity_id,
            close_period,
            difficulty,
            ood,
        )
        latent_regime = variance.regime.value
    elif task == TaskId.CASH_RECONCILIATION:
        cash = _build_cash(
            h,
            identity,
            entity_id,
            close_period,
            difficulty,
            ood,
        )
        latent_regime = cash.regime.value
    else:
        raise ValueError(f"unsupported finance task {task}")

    return LatentWorld(
        world_id=world_id,
        group_id=group_id,
        entity_id=entity_id,
        close_period=close_period,
        chart_of_accounts=chart,
        vendors=vendors,
        policies=policies,
        departments=("operations", "engineering", "finance", "sales", "treasury"),
        transaction=transaction,
        variance=variance,
        cash=cash,
        latent_regime=latent_regime,
        ood_held_out=ood,
        metadata={
            "seed": seed,
            "index": index,
            "salt": salt,
            "split_domain": split_token,
        },
    )


def _build_chart(group_identity: str, *, ood: bool) -> tuple[GLAccount, ...]:
    family = "schedule" if not ood else "crosswalk"
    return tuple(
        GLAccount(
            code,
            f"{base} — {_natural_name(f'{group_identity}|coa|{code}|{family}')}",
            account_type,
        )
        for code, base, account_type in CHART_ACCOUNT_SPECS
    )


def _build_vendors(group_identity: str, *, ood: bool) -> tuple[Vendor, ...]:
    archetypes = OOD_VENDOR_ARCHETYPES if ood else IID_VENDOR_ARCHETYPES
    vendors: list[Vendor] = []
    for archetype in archetypes:
        key = f"{group_identity}|vendor|{archetype.key}"
        name = _natural_name(key)
        if archetype.key == "consulting":
            name = f"Deloitte {name}"
        corruption = _natural_name(f"{key}|descriptor").replace(" ", "-").casefold()
        descriptors = (
            f"{corruption} {name} {archetype.descriptor_term}",
            f"{name} {archetype.descriptor_term} ref {corruption}",
        )
        vendors.append(
            Vendor(
                vendor_id=_opaque_id("vnd_", key),
                archetype=archetype.key,
                name=name,
                descriptors=descriptors,
                default_gl=archetype.default_gl,
                category=archetype.category,
                corruption_family=corruption,
            )
        )
    return tuple(vendors)


def _build_policies(
    group_identity: str,
    vendors: Sequence[Vendor],
    *,
    ood: bool,
) -> tuple[PolicyRule, ...]:
    by_category: dict[str, tuple[str, ...]] = {}
    for category in {vendor.category for vendor in vendors}:
        by_category[category] = tuple(
            vendor.vendor_id for vendor in vendors if vendor.category == category
        )

    specs: list[
        tuple[
            str,
            int,
            PolicyAction,
            str,
            int,
            int | None,
            tuple[str, ...],
            str,
        ]
    ] = [
        ("MEAL-LOW", 30, PolicyAction.APPROVE, "6100", 0, 5_000, ("meal", "cafe"), "meals"),
        (
            "MEAL-MID",
            20,
            PolicyAction.REVIEW,
            "6100",
            5_001,
            15_000,
            ("meal", "entertainment"),
            "meals",
        ),
        (
            "MEAL-HIGH",
            10,
            PolicyAction.REJECT,
            "6100",
            15_001,
            None,
            ("meal", "entertainment"),
            "meals",
        ),
        (
            "AIR-BASE",
            40,
            PolicyAction.APPROVE,
            "6205",
            0,
            500_000,
            ("air travel", "flight"),
            "airfare",
        ),
        (
            "AIR-HIGH",
            35,
            PolicyAction.REVIEW,
            "6205",
            500_001,
            None,
            ("air travel", "flight"),
            "airfare",
        ),
        (
            "LODGE-BASE",
            40,
            PolicyAction.REVIEW,
            "6210",
            0,
            300_000,
            ("hotel", "lodging", "inn"),
            "lodging",
        ),
        (
            "LODGE-HIGH",
            35,
            PolicyAction.REJECT,
            "6210",
            300_001,
            None,
            ("hotel", "lodging", "inn"),
            "lodging",
        ),
        (
            "SAAS-BASE",
            50,
            PolicyAction.APPROVE,
            "6400",
            0,
            200_000,
            ("software", "subscription"),
            "saas",
        ),
        (
            "SAAS-HIGH",
            45,
            PolicyAction.REVIEW,
            "6400",
            200_001,
            None,
            ("software", "subscription"),
            "saas",
        ),
        (
            "CLOUD-BASE",
            50,
            PolicyAction.APPROVE if not ood else PolicyAction.REVIEW,
            "6500",
            0,
            2_000_000,
            ("cloud", "compute", "capacity"),
            "cloud",
        ),
        (
            "CLOUD-HIGH",
            45,
            PolicyAction.REVIEW,
            "6500",
            2_000_001,
            None,
            ("cloud", "compute", "capacity"),
            "cloud",
        ),
        (
            "CAPEX",
            5,
            PolicyAction.REJECT,
            "1500",
            0,
            None,
            ("capital equipment", "machinery", "server"),
            "capex",
        ),
        (
            "IT",
            60,
            PolicyAction.APPROVE,
            "6400",
            0,
            1_000_000,
            ("it", "laptop hardware"),
            "it",
        ),
        (
            "PERSONAL",
            8,
            PolicyAction.REJECT,
            "6100",
            0,
            None,
            ("personal", "gift card"),
            "personal",
        ),
        (
            "SERVICES",
            40,
            PolicyAction.REVIEW,
            "6600",
            0,
            None,
            ("consulting", "advisory", "professional services"),
            "services",
        ),
        (
            "FEES",
            40,
            PolicyAction.APPROVE,
            "6700",
            0,
            None,
            ("processing fee", "bank fee"),
            "fees",
        ),
        (
            "FACILITIES",
            40,
            PolicyAction.APPROVE,
            "6400",
            0,
            100_000,
            ("workspace", "desk"),
            "facilities",
        ),
        (
            "FACILITIES-HIGH",
            35,
            PolicyAction.REVIEW,
            "6400",
            100_001,
            None,
            ("workspace", "desk"),
            "facilities",
        ),
    ]

    context = _natural_name(f"{group_identity}|policy-context")
    rules: list[PolicyRule] = []
    for (
        semantic_code,
        precedence,
        action,
        gl_account,
        min_amount,
        max_amount,
        keywords,
        category,
    ) in specs:
        if category not in by_category and category != "it":
            continue
        unique_code = semantic_code
        suffix = _digest(f"{group_identity}|policy|{unique_code}")[:8].upper()
        rule_id = f"POL-{unique_code}-{suffix}"
        upper = "unbounded" if max_amount is None else str(max_amount)
        text = (
            f"{context} control for {category}: amounts {min_amount} through {upper} "
            f"use account {gl_account} with decision {action.value}; rule {rule_id}."
        )
        rules.append(
            PolicyRule(
                rule_id=rule_id,
                semantic_code=unique_code,
                precedence=precedence,
                action=action,
                gl_account=gl_account,
                min_amount_minor=min_amount,
                max_amount_minor=max_amount,
                keywords=keywords,
                category=category,
                vendor_ids=by_category.get(category, ()),
                text=text,
            )
        )
    return tuple(sorted(rules, key=lambda rule: (rule.precedence, rule.rule_id)))


def _build_transaction(
    h: int,
    identity: str,
    vendors: Sequence[Vendor],
    policies: Sequence[PolicyRule],
    entity_id: str,
    difficulty: Difficulty,
) -> TransactionLatent:
    vendor = vendors[h % len(vendors)]
    hard = TxnHardNegative.NONE
    if difficulty == Difficulty.MEDIUM:
        hard = _stable_choice(
            h >> 4,
            (
                TxnHardNegative.NEAR_SYNONYM_GL,
                TxnHardNegative.THRESHOLD_BOUNDARY,
                TxnHardNegative.MISLEADING_DESCRIPTOR,
                TxnHardNegative.REFUND,
                TxnHardNegative.NONE,
            ),
        )
    elif difficulty == Difficulty.HARD:
        hard = _stable_choice(
            h >> 4,
            (
                TxnHardNegative.CONFLICTING_RULES,
                TxnHardNegative.CAPEX_OPEX,
                TxnHardNegative.THRESHOLD_BOUNDARY,
                TxnHardNegative.MISLEADING_DESCRIPTOR,
                TxnHardNegative.ALLOWED_LOOKING_PROHIBITED,
                TxnHardNegative.NEAR_SYNONYM_GL,
                TxnHardNegative.SPLIT_ALLOCATION,
                TxnHardNegative.REFUND,
                TxnHardNegative.CHARGEBACK,
                TxnHardNegative.REFUND_SPLIT,
                TxnHardNegative.PERSONAL_LOOKING_ALLOWED,
            ),
        )

    if hard in {TxnHardNegative.CAPEX_OPEX, TxnHardNegative.CONFLICTING_RULES}:
        vendor = _vendor_for_category(vendors, "capex")
    elif hard in {TxnHardNegative.SPLIT_ALLOCATION, TxnHardNegative.REFUND_SPLIT}:
        vendor = _vendor_for_category(vendors, "services")
    elif hard in {
        TxnHardNegative.THRESHOLD_BOUNDARY,
        TxnHardNegative.NEAR_SYNONYM_GL,
        TxnHardNegative.PERSONAL_LOOKING_ALLOWED,
    }:
        vendor = _vendor_for_category(vendors, "meals")
    elif hard == TxnHardNegative.ALLOWED_LOOKING_PROHIBITED:
        vendor = _vendor_for_category(vendors, "personal")

    amount = 1_001 + (h % 1_900_000)
    if vendor.category == "meals":
        amount = 1_001 + (h % 30_000)
    if hard == TxnHardNegative.THRESHOLD_BOUNDARY:
        amount = (5_000, 5_001, 15_000, 15_001)[(h >> 8) % 4]
    if hard in {TxnHardNegative.CAPEX_OPEX, TxnHardNegative.CONFLICTING_RULES}:
        amount = 1_500_000 + (h % 4_000_000)

    descriptor_context = _natural_name(f"{identity}|descriptor-context")
    descriptor = (
        f"{vendor.descriptors[h % len(vendor.descriptors)]} merchant reference {descriptor_context}"
    )
    if hard == TxnHardNegative.MISLEADING_DESCRIPTOR:
        descriptor = f"{descriptor} consulting retainer"
    elif hard == TxnHardNegative.PERSONAL_LOOKING_ALLOWED:
        descriptor = (
            f"{vendor.corruption_family} {descriptor_context} "
            f"gift card style team meal {vendor.name}"
        )
    elif hard == TxnHardNegative.ALLOWED_LOOKING_PROHIBITED:
        descriptor = (
            f"{vendor.corruption_family} {descriptor_context} "
            f"ordinary office order "
            f"{vendor.name} personal"
        )
    elif hard == TxnHardNegative.CONFLICTING_RULES:
        descriptor = f"{descriptor} server laptop hardware"

    transaction_kind: Literal["charge", "refund", "chargeback"] = "charge"
    if hard in {TxnHardNegative.REFUND, TxnHardNegative.REFUND_SPLIT}:
        transaction_kind = "refund"
        descriptor = f"refund {descriptor}"
        amount = -abs(amount)
    elif hard == TxnHardNegative.CHARGEBACK:
        transaction_kind = "chargeback"
        descriptor = f"chargeback reversal {descriptor}"
        amount = -abs(amount)

    action, gl_account, rule_ids = resolve_policy(
        policies,
        descriptor=descriptor,
        amount_minor=amount,
        category=vendor.category,
        vendor_id=vendor.vendor_id,
    )

    split_lines: tuple[tuple[str, int], ...] = ()
    if hard in {TxnHardNegative.SPLIT_ALLOCATION, TxnHardNegative.REFUND_SPLIT}:
        total = abs(amount)
        first = total // 2 + (h % 17)
        if first >= total:
            first = total // 2
        split_lines = ((gl_account, first), ("6400", total - first))

    month = (h % 12) + 1
    day = ((h >> 6) % 27) + 1
    return TransactionLatent(
        txn_id=_opaque_id("txn_", f"{identity}|transaction"),
        vendor_id=vendor.vendor_id,
        descriptor=descriptor,
        amount_minor=amount,
        currency="USD",
        date=f"2026-{month:02d}-{day:02d}",
        entity_id=entity_id,
        cost_center=("operations", "engineering", "finance", "sales")[h % 4],
        transaction_kind=transaction_kind,
        hard_negative=hard,
        gl_account=gl_account,
        policy_action=action,
        applied_rule_ids=rule_ids,
        split_lines=split_lines,
    )


def _build_variance(
    h: int,
    identity: str,
    entity_id: str,
    period: str,
    difficulty: Difficulty,
    ood: bool,
) -> VarianceLatent:
    if difficulty == Difficulty.EASY:
        regime = VarianceRegime.SIMPLE
    elif difficulty == Difficulty.MEDIUM:
        regime = _stable_choice(
            h,
            (
                VarianceRegime.SIMPLE,
                VarianceRegime.OFFSET,
                VarianceRegime.PRICE_VOLUME,
            ),
        )
    else:
        regime = _stable_choice(
            h,
            (
                VarianceRegime.OFFSET,
                VarianceRegime.PRICE_VOLUME,
                VarianceRegime.FX,
                VarianceRegime.TIE,
                VarianceRegime.HIDDEN_SUBTOTAL,
            ),
        )
    if ood:
        regime = _stable_choice(
            h >> 3,
            (
                VarianceRegime.OFFSET,
                VarianceRegime.PRICE_VOLUME,
                VarianceRegime.FX,
            ),
        )

    def observation(
        driver_id: str,
        pnl_type: Literal["expense", "revenue"],
        budget: int,
        actual: int,
        kind: Literal["expense", "revenue", "fx", "volume", "price", "other"],
    ) -> VarianceObservation:
        return VarianceObservation(
            source_id=_opaque_id("src_", f"{identity}|variance|{driver_id}"),
            driver_id=driver_id,
            pnl_type=pnl_type,
            budget_minor=budget,
            actual_minor=actual,
            kind=kind,
        )

    base = 600_000 + (h % 300_000)
    unallocated: tuple[VarianceObservation, ...] = ()
    if regime == VarianceRegime.SIMPLE:
        delta = 40_001 + (h % 80_000)
        drivers = (observation("staffing", "expense", base, base - delta, "expense"),)
    elif regime == VarianceRegime.OFFSET:
        expense_budget = 300_000 + (h % 70_000)
        expense_over = 120_001 + (h % 60_000)
        revenue_budget = 90_000 + (h % 30_000)
        revenue_gain = 40_001 + (h % 50_000)
        names = ("capacity", "renewals") if not ood else ("fleet_mix", "channel_yield")
        drivers = (
            observation(
                names[0],
                "expense",
                expense_budget,
                expense_budget + expense_over,
                "expense",
            ),
            observation(
                names[1],
                "revenue",
                revenue_budget,
                revenue_budget + revenue_gain,
                "revenue",
            ),
        )
    elif regime == VarianceRegime.PRICE_VOLUME:
        budget_units = 80 + (h % 21)
        actual_units = budget_units + 5 + ((h >> 5) % 13)
        budget_rate = 1_100 + ((h >> 9) % 400)
        actual_rate = budget_rate + 20 + ((h >> 13) % 80)
        names = ("unit_price", "usage_volume") if not ood else ("mix_rate", "load_count")
        drivers = (
            observation(
                names[0],
                "expense",
                actual_units * budget_rate,
                actual_units * actual_rate,
                "price",
            ),
            observation(
                names[1],
                "expense",
                budget_units * budget_rate,
                actual_units * budget_rate,
                "volume",
            ),
        )
        base = budget_units * budget_rate
    elif regime == VarianceRegime.FX:
        names = (
            ("volume", "price", "currency")
            if not ood
            else ("international_units", "contract_rate", "translation")
        )
        amounts = (
            (150_000 + (h % 20_000), 210_000 + (h % 20_000)),
            (180_000 + (h % 25_000), 145_000 + (h % 25_000)),
            (80_000 + (h % 10_000), 105_000 + (h % 10_000)),
        )
        drivers = (
            observation(names[0], "expense", *amounts[0], "volume"),
            observation(names[1], "expense", *amounts[1], "price"),
            observation(names[2], "expense", *amounts[2], "fx"),
        )
    elif regime == VarianceRegime.TIE:
        magnitude = 90_001 + (h % 20_000)
        drivers = (
            observation("alpha_cost", "expense", base, base + magnitude, "expense"),
            observation("beta_cost", "expense", base + 7, base + 7 + magnitude, "expense"),
        )
    else:
        cloud_budget = 250_000 + (h % 30_000)
        support_budget = 100_000 + (h % 20_000)
        hidden_budget = 50_000 + (h % 10_000)
        drivers = (
            observation(
                "cloud_usage",
                "expense",
                cloud_budget,
                cloud_budget + 180_000,
                "expense",
            ),
            observation(
                "support_volume",
                "expense",
                support_budget,
                support_budget + 70_000,
                "expense",
            ),
        )
        unallocated = (
            observation(
                "unallocated_close_items",
                "expense",
                hidden_budget,
                hidden_budget + 30_000,
                "other",
            ),
        )

    all_observations = drivers + unallocated
    baseline = 400_000 + ((h >> 17) % 200_000)
    budget = baseline + sum(
        obs.budget_minor if obs.pnl_type == "expense" else -obs.budget_minor
        for obs in all_observations
    )
    actual = baseline + sum(
        obs.actual_minor if obs.pnl_type == "expense" else -obs.actual_minor
        for obs in all_observations
    )

    context = _natural_name(f"{identity}|variance-rule")
    mat_id = f"VAR-MAT-{_digest(f'{identity}|mat')[:8].upper()}"
    rules = [(mat_id, f"{context} materiality uses exact minor-unit impacts.")]
    rule_ids = [mat_id]
    if regime == VarianceRegime.TIE:
        tie_id = f"VAR-TIE-{_digest(f'{identity}|tie')[:8].upper()}"
        rules.append((tie_id, f"{context} ties sort by driver identifier ascending."))
        rule_ids = [tie_id]
    if regime == VarianceRegime.FX:
        fx_id = f"VAR-FX-{_digest(f'{identity}|fx')[:8].upper()}"
        rules.append((fx_id, f"{context} reports translation separately."))
        rule_ids.append(fx_id)

    return VarianceLatent(
        period=period,
        budget_minor=budget,
        actual_minor=actual,
        drivers=drivers,
        unallocated=unallocated,
        rule_ids=tuple(rule_ids),
        analysis_rules=tuple(rules),
        regime=regime,
        entity_id=entity_id,
    )


def _build_cash(
    h: int,
    identity: str,
    entity_id: str,
    close_period: str,
    difficulty: Difficulty,
    ood: bool,
) -> CashLatent:
    if ood:
        regime = _stable_choice(
            h >> 2,
            (
                CashRegime.ONE_TO_MANY,
                CashRegime.MANY_TO_ONE,
                CashRegime.PARTIAL,
                CashRegime.SAME_AMOUNT_COLLISION,
            ),
        )
    elif difficulty == Difficulty.EASY:
        regime = CashRegime.CLEAN_MATCH
    elif difficulty == Difficulty.MEDIUM:
        regime = _stable_choice(
            h,
            (
                CashRegime.CLEAN_MATCH,
                CashRegime.BANK_FEE,
                CashRegime.DEPOSIT_IN_TRANSIT,
            ),
        )
    else:
        regime = _stable_choice(
            h,
            (
                CashRegime.BANK_FEE,
                CashRegime.DEPOSIT_IN_TRANSIT,
                CashRegime.STALE_CHECK,
                CashRegime.DUPLICATE,
            ),
        )

    def book(suffix: str, amount: int, day: int, memo: str) -> BookEntry:
        return BookEntry(
            _opaque_id("bok_", f"{identity}|book|{suffix}"),
            amount,
            f"2026-03-{day:02d}",
            memo,
        )

    def bank(
        suffix: str,
        amount: int,
        day: int,
        memo: str,
        event_type: Literal["clearing", "fee", "duplicate", "other"] = "clearing",
    ) -> BankEvent:
        return BankEvent(
            _opaque_id("bnk_", f"{identity}|bank|{suffix}"),
            amount,
            f"2026-03-{day:02d}",
            memo,
            event_type,
        )

    base = 7_000_000 + (h % 900_000)
    first_amount = 200_001 + ((h >> 8) % 90_000)
    b1 = book("primary", first_amount, 1, "customer remittance")
    k1 = bank("primary", first_amount, 2, "cleared remittance")
    book_entries = [b1]
    bank_events = [k1]
    matched: list[tuple[tuple[str, ...], tuple[str, ...]]] = [((b1.entry_id,), (k1.event_id,))]
    exceptions: list[tuple[str, tuple[str, ...], int]] = []
    book_balance = base
    bank_balance = base

    adjustment = 3_001 + ((h >> 11) % 40_000)
    if regime == CashRegime.BANK_FEE:
        fee = bank("fee", -adjustment, 3, "service charge", "fee")
        bank_events.append(fee)
        exceptions.append(("bank_fee", (fee.event_id,), adjustment))
        bank_balance -= adjustment
    elif regime == CashRegime.DEPOSIT_IN_TRANSIT:
        b2 = book("transit", adjustment, 4, "deposit in transit")
        book_entries.append(b2)
        exceptions.append(("deposit_in_transit", (b2.entry_id,), adjustment))
        book_balance += adjustment
    elif regime == CashRegime.STALE_CHECK:
        b2 = book("stale", -adjustment, 5, "stale check")
        book_entries.append(b2)
        exceptions.append(("stale_check", (b2.entry_id,), adjustment))
        book_balance -= adjustment
    elif regime == CashRegime.DUPLICATE:
        duplicate = bank("duplicate", first_amount, 2, "duplicate remittance", "duplicate")
        bank_events.append(duplicate)
        exceptions.append(("duplicate", (duplicate.event_id,), first_amount))
        bank_balance += first_amount
    elif regime == CashRegime.ONE_TO_MANY:
        total = 90_001 + ((h >> 7) % 30_000)
        left = total // 2
        b2 = book("aggregate", total, 6, "aggregate receipt")
        k2 = bank("aggregate-a", left, 6, "settlement part alpha")
        k3 = bank("aggregate-b", total - left, 7, "settlement part beta")
        book_entries.append(b2)
        bank_events.extend((k2, k3))
        matched.append(((b2.entry_id,), (k2.event_id, k3.event_id)))
        book_balance += total
        bank_balance += total
    elif regime == CashRegime.MANY_TO_ONE:
        left = 40_001 + ((h >> 7) % 20_000)
        right = 30_001 + ((h >> 12) % 20_000)
        b2 = book("bundle-a", left, 6, "invoice alpha")
        b3 = book("bundle-b", right, 7, "invoice beta")
        k2 = bank("bundle", left + right, 8, "combined settlement")
        book_entries.extend((b2, b3))
        bank_events.append(k2)
        matched.append(((b2.entry_id, b3.entry_id), (k2.event_id,)))
        book_balance += left + right
        bank_balance += left + right
    elif regime == CashRegime.PARTIAL:
        booked = 80_001 + ((h >> 7) % 30_000)
        settled = booked - adjustment
        b2 = book("partial-book", booked, 8, "invoice awaiting remainder")
        k2 = bank("partial-bank", settled, 9, "partial settlement")
        book_entries.append(b2)
        bank_events.append(k2)
        exceptions.append(("partial_settlement", (b2.entry_id, k2.event_id), adjustment))
        book_balance += booked
        bank_balance += settled
    elif regime == CashRegime.SAME_AMOUNT_COLLISION:
        amount = 60_001 + ((h >> 7) % 20_000)
        b2 = book("collision-a", amount, 10, "customer alpha")
        b3 = book("collision-b", amount, 10, "customer beta")
        k2 = bank("collision-a", amount, 11, "alpha settlement")
        k3 = bank("collision-b", amount, 11, "beta settlement")
        book_entries.extend((b2, b3))
        bank_events.extend((k2, k3))
        matched.extend(
            (
                ((b2.entry_id,), (k2.event_id,)),
                ((b3.entry_id,), (k3.event_id,)),
            )
        )
        book_balance += amount * 2
        bank_balance += amount * 2

    return CashLatent(
        book_balance_minor=book_balance,
        bank_balance_minor=bank_balance,
        book_entries=tuple(book_entries),
        bank_events=tuple(bank_events),
        matched=tuple(matched),
        exceptions=tuple(exceptions),
        regime=regime,
        entity_id=entity_id,
        close_period=close_period,
    )


def _vendor_for_category(vendors: Sequence[Vendor], category: str) -> Vendor:
    try:
        return next(vendor for vendor in vendors if vendor.category == category)
    except StopIteration as exc:
        raise ValueError(f"missing vendor category {category!r}") from exc


def _stable_choice(value: int, options: Sequence[Any]) -> Any:
    return options[value % len(options)]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _hash_int(value: str) -> int:
    return int(_digest(value)[:16], 16)


def _opaque_id(prefix: str, value: str) -> str:
    return f"{prefix}{_digest(value)[:18]}"


def _natural_name(value: str) -> str:
    digest = bytes.fromhex(_digest(value))
    coined = "".join(chr(ord("a") + byte % 26) for byte in digest[3:10]).title()
    return (
        f"{_NAME_LEFT[digest[0] % len(_NAME_LEFT)]} "
        f"{_NAME_LEFT[digest[1] % len(_NAME_LEFT)]} "
        f"{_NAME_RIGHT[digest[2] % len(_NAME_RIGHT)]} {coined}"
    )


def _period_label(value: str) -> str:
    digest = bytes.fromhex(_digest(f"{value}|period"))
    year = 2025 + int.from_bytes(digest[:2], "big") % 60
    period = 1 + digest[2] % 13
    context = _NAME_LEFT[digest[3] % len(_NAME_LEFT)]
    coined = "".join(chr(ord("A") + byte % 26) for byte in digest[4:11])
    return f"FY{year}-P{period:02d}-{context}-{coined}"
