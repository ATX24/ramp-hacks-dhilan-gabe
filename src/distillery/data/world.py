"""Deterministic latent finance world state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal

from distillery.contracts.hashing import content_sha256
from distillery.contracts.tasks import Difficulty, TaskId


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
    PARTIAL = "partial"


class TxnHardNegative(StrEnum):
    NONE = "none"
    NEAR_SYNONYM_GL = "near_synonym_gl"
    REFUND = "refund"
    CAPEX_OPEX = "capex_opex"
    SPLIT_ALLOCATION = "split_allocation"
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
    precedence: int  # lower wins
    action: PolicyAction
    gl_account: str
    max_amount_minor: int | None
    keywords: tuple[str, ...]
    category: str


@dataclass(frozen=True)
class Vendor:
    vendor_id: str
    name: str
    descriptors: tuple[str, ...]
    default_gl: str
    category: str


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
    is_refund: bool
    hard_negative: TxnHardNegative
    # Oracle-resolved fields (set at world build).
    gl_account: str
    policy_action: PolicyAction
    applied_rule_ids: tuple[str, ...]
    contra_account: str = "2100"
    split_lines: tuple[tuple[str, int], ...] = ()  # (account, amount_minor)


@dataclass(frozen=True)
class VarianceDriverLatent:
    driver_id: str
    impact_minor: int
    evidence_id: str
    kind: Literal["expense", "revenue", "fx", "volume", "price"]


@dataclass(frozen=True)
class VarianceLatent:
    period: str
    budget_minor: int
    actual_minor: int
    drivers: tuple[VarianceDriverLatent, ...]
    other_impact_minor: int
    materiality_rule_id: str
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
    exceptions: tuple[tuple[str, tuple[str, ...], int], ...]  # type, ids, amount
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


# --- Catalogs (deterministic, no I/O) ---

CHART: tuple[GLAccount, ...] = (
    GLAccount("1500", "PPE / Capex", "asset"),
    GLAccount("2100", "Accounts Payable", "liability"),
    GLAccount("4000", "Revenue", "revenue"),
    GLAccount("6100", "Meals & Entertainment", "expense"),
    GLAccount("6105", "Office Refreshments", "expense"),  # near-synonym of 6100
    GLAccount("6205", "Travel Airfare", "expense"),
    GLAccount("6210", "Travel Lodging", "expense"),
    GLAccount("6400", "Software & SaaS", "expense"),
    GLAccount("6500", "Cloud Infrastructure", "expense"),
    GLAccount("6600", "Professional Services", "expense"),
    GLAccount("6700", "Bank Fees", "expense"),
)

IID_VENDORS: tuple[Vendor, ...] = (
    Vendor("vnd_starbucks", "Starbucks", ("SQ *STARBUCKS", "SBUX"), "6100", "meals"),
    Vendor("vnd_united", "United Airlines", ("UNITED", "UA "), "6205", "travel"),
    Vendor("vnd_aws", "Amazon Web Services", ("AWS", "AMAZON WEB SERVICES"), "6500", "cloud"),
    Vendor("vnd_dell", "Dell Technologies", ("DELL", "DELL TECHNOLOGIES"), "1500", "capex"),
    Vendor("vnd_slack", "Slack Technologies", ("SLACK", "SLACK T"), "6400", "saas"),
    Vendor("vnd_uber", "Uber", ("UBER", "UBER EATS"), "6100", "meals"),
    Vendor("vnd_hilton", "Hilton", ("HILTON", "HI HOTELS"), "6210", "travel"),
    Vendor("vnd_deloitte", "Deloitte", ("DELOITTE", "DELOITTE CONSULTING"), "6600", "services"),
)

OOD_VENDORS: tuple[Vendor, ...] = (
    Vendor("vnd_wework", "WeWork", ("WEWORK", "WW DESK"), "6400", "saas"),
    Vendor("vnd_stripe", "Stripe", ("STRIPE", "STRIPE FEE"), "6700", "fees"),
    Vendor("vnd_gcp", "Google Cloud", ("GCP", "GOOGLE CLOUD"), "6500", "cloud"),
    Vendor("vnd_marriott", "Marriott", ("MARRIOTT", "MC HOTELS"), "6210", "travel"),
)

IID_POLICIES: tuple[PolicyRule, ...] = (
    PolicyRule(
        "POL-MEAL-001", 100, PolicyAction.APPROVE, "6100", 5000,
        ("meal", "starbucks", "uber"), "meals",
    ),
    PolicyRule(
        "POL-MEAL-002", 90, PolicyAction.REVIEW, "6100", 15000,
        ("meal", "entertainment"), "meals",
    ),
    PolicyRule(
        "POL-TRAVEL-017", 80, PolicyAction.APPROVE, "6205", 500000,
        ("airfare", "united", "flight"), "travel",
    ),
    PolicyRule(
        "POL-TRAVEL-018", 85, PolicyAction.REVIEW, "6210", 300000,
        ("hotel", "hilton", "lodging"), "travel",
    ),
    PolicyRule(
        "POL-SAAS-003", 70, PolicyAction.APPROVE, "6400", 200000,
        ("saas", "slack", "software"), "saas",
    ),
    PolicyRule(
        "POL-CLOUD-004", 60, PolicyAction.APPROVE, "6500", 2_000_000,
        ("cloud", "aws", "infra"), "cloud",
    ),
    PolicyRule(
        "POL-CAPEX-009", 10, PolicyAction.REJECT, "1500", None,
        ("server", "dell", "ppe", "capex"), "capex",
    ),
    PolicyRule(
        "POL-IT-002", 50, PolicyAction.APPROVE, "6400", 1_000_000,
        ("laptop", "hardware", "it"), "it",
    ),
    PolicyRule(
        "POL-PERSONAL-011", 20, PolicyAction.REJECT, "6100", None,
        ("personal", "gift card"), "personal",
    ),
    PolicyRule(
        "POL-SVC-006", 75, PolicyAction.REVIEW, "6600", 500000,
        ("consulting", "deloitte"), "services",
    ),
)

OOD_POLICIES: tuple[PolicyRule, ...] = (
    PolicyRule(
        "POL-OOD-DESK-101", 40, PolicyAction.APPROVE, "6400", 100000,
        ("wework", "desk"), "facilities",
    ),
    PolicyRule(
        "POL-OOD-FEE-102", 30, PolicyAction.APPROVE, "6700", 50000,
        ("stripe", "fee"), "fees",
    ),
    PolicyRule(
        "POL-OOD-CLOUD-103", 55, PolicyAction.REVIEW, "6500", 5_000_000,
        ("gcp", "google cloud"), "cloud",
    ),
    PolicyRule(
        "POL-OOD-TRAVEL-104", 65, PolicyAction.APPROVE, "6210", 400000,
        ("marriott", "hotel"), "travel",
    ),
)

IID_TEMPLATE_FAMILIES: dict[TaskId, tuple[str, ...]] = {
    TaskId.TRANSACTION_REVIEW: (
        "txn_meal_v1",
        "txn_policy_v3",
        "txn_capex_v2",
        "txn_saas_v1",
        "txn_conflict_v1",
    ),
    TaskId.VARIANCE_ANALYSIS: (
        "var_simple_v1",
        "var_drivers_v2",
        "var_offset_v1",
        "var_tie_v1",
    ),
    TaskId.CASH_RECONCILIATION: (
        "cash_match_v1",
        "cash_exceptions_v1",
        "cash_partial_v1",
    ),
}

OOD_TEMPLATE_FAMILIES: dict[TaskId, tuple[str, ...]] = {
    TaskId.TRANSACTION_REVIEW: (
        "txn_ood_policy_mix_v1",
        "txn_ood_gl_desc_v1",
        "txn_ood_threshold_v1",
    ),
    TaskId.VARIANCE_ANALYSIS: (
        "var_ood_driver_mix_v1",
        "var_ood_sign_v1",
        "var_ood_fx_v1",
    ),
    TaskId.CASH_RECONCILIATION: (
        "cash_ood_agg_v1",
        "cash_ood_collision_v1",
    ),
}


def _stable_choice(rng_value: int, options: Sequence[Any]) -> Any:
    return options[rng_value % len(options)]


def resolve_policy(
    policies: tuple[PolicyRule, ...],
    *,
    descriptor: str,
    amount_minor: int,
    category: str,
    hard_negative: TxnHardNegative,
) -> tuple[PolicyAction, str, tuple[str, ...]]:
    """Apply explicit precedence: lowest precedence number wins among matches."""
    text = descriptor.lower()
    matches: list[PolicyRule] = []
    for rule in policies:
        keyword_hit = any(k in text for k in rule.keywords) or rule.category == category
        if not keyword_hit:
            continue
        if rule.max_amount_minor is not None and amount_minor > rule.max_amount_minor:
            # threshold boundary hard-negatives still match but escalate action
            if hard_negative == TxnHardNegative.THRESHOLD_BOUNDARY:
                matches.append(rule)
            continue
        matches.append(rule)

    if hard_negative == TxnHardNegative.CONFLICTING_RULES:
        forced_ids = {"POL-CAPEX-009", "POL-IT-002", "POL-OOD-DESK-101", "POL-OOD-CLOUD-103"}
        forced = [p for p in policies if p.rule_id in forced_ids]
        matches = list({m.rule_id: m for m in (matches + forced)}.values())

    if hard_negative == TxnHardNegative.ALLOWED_LOOKING_PROHIBITED:
        personal = next((p for p in policies if p.rule_id == "POL-PERSONAL-011"), None)
        if personal is None:
            personal = next(
                (p for p in policies if p.action == PolicyAction.REJECT),
                None,
            )
        if personal is not None:
            matches = [personal]

    if hard_negative == TxnHardNegative.PERSONAL_LOOKING_ALLOWED:
        meal = next((p for p in policies if p.rule_id == "POL-MEAL-001"), None)
        if meal is None:
            meal = next((p for p in policies if p.action == PolicyAction.APPROVE), None)
        if meal is not None:
            matches = [meal]

    if not matches:
        fallback = next((p for p in policies if p.action == PolicyAction.REVIEW), policies[0])
        return fallback.action, fallback.gl_account, (fallback.rule_id,)

    matches.sort(key=lambda r: (r.precedence, r.rule_id))
    winner = matches[0]
    action = winner.action
    gl = winner.gl_account

    if hard_negative == TxnHardNegative.THRESHOLD_BOUNDARY and winner.max_amount_minor is not None:
        if amount_minor > winner.max_amount_minor:
            action = (
                PolicyAction.REJECT
                if winner.action != PolicyAction.REJECT
                else PolicyAction.REVIEW
            )

    if hard_negative == TxnHardNegative.CAPEX_OPEX:
        capex = next((p for p in policies if p.rule_id == "POL-CAPEX-009"), None)
        if capex is None:
            capex = next((p for p in policies if p.gl_account == "1500"), winner)
        gl = capex.gl_account
        action = PolicyAction.REJECT
        winner = capex

    return action, gl, (winner.rule_id,)


def build_world(
    *,
    seed: int,
    index: int,
    split_token: str,
    task: TaskId,
    difficulty: Difficulty,
    ood: bool = False,
) -> LatentWorld:
    """Build a deterministic latent world for one example slot."""
    # Mix seeds so (seed, index, split, task, difficulty, ood) is unique.
    h = _mix(seed, index, _fnv(split_token), _fnv(task.value), _fnv(difficulty.value), int(ood))
    entity_id = f"ent_{split_token}_{index:05d}"
    world_id = f"world_{split_token}_{index:05d}"
    group_id = f"grp_{split_token}_{index // 4:05d}"
    close_period = f"2026-Q{(h % 4) + 1}"

    vendors = OOD_VENDORS if ood else IID_VENDORS
    policies = tuple(sorted(IID_POLICIES + (OOD_POLICIES if ood else ()), key=lambda p: p.rule_id))
    if ood:
        # OOD holds out novel policy combinations: use OOD-only policies for matching.
        policies = OOD_POLICIES + tuple(p for p in IID_POLICIES if p.rule_id.startswith("POL-MEAL"))

    departments = ("ops", "eng", "finance", "sales", "treasury")
    latent_regime = "ood" if ood else "iid"

    txn: TransactionLatent | None = None
    variance: VarianceLatent | None = None
    cash: CashLatent | None = None

    if task == TaskId.TRANSACTION_REVIEW:
        txn = _build_transaction(h, vendors, policies, entity_id, difficulty, ood)
        if txn.hard_negative != TxnHardNegative.NONE:
            latent_regime = txn.hard_negative.value
        else:
            latent_regime = "ood_txn" if ood else "iid_txn"
    elif task == TaskId.VARIANCE_ANALYSIS:
        variance = _build_variance(h, entity_id, difficulty, ood)
        latent_regime = variance.regime.value
    elif task == TaskId.CASH_RECONCILIATION:
        cash = _build_cash(h, entity_id, close_period, difficulty, ood)
        latent_regime = cash.regime.value
    else:
        raise ValueError(f"unsupported task for finance world generator: {task}")

    return LatentWorld(
        world_id=world_id,
        group_id=group_id,
        entity_id=entity_id,
        close_period=close_period,
        chart_of_accounts=CHART,
        vendors=vendors,
        policies=policies,
        departments=departments,
        transaction=txn,
        variance=variance,
        cash=cash,
        latent_regime=latent_regime,
        ood_held_out=ood,
        metadata={"seed": seed, "index": index, "split_token": split_token},
    )


def _build_transaction(
    h: int,
    vendors: tuple[Vendor, ...],
    policies: tuple[PolicyRule, ...],
    entity_id: str,
    difficulty: Difficulty,
    ood: bool,
) -> TransactionLatent:
    vendor = _stable_choice(h, vendors)
    hard = TxnHardNegative.NONE
    if difficulty == Difficulty.HARD:
        options = (
            TxnHardNegative.CONFLICTING_RULES,
            TxnHardNegative.CAPEX_OPEX,
            TxnHardNegative.THRESHOLD_BOUNDARY,
            TxnHardNegative.MISLEADING_DESCRIPTOR,
            TxnHardNegative.ALLOWED_LOOKING_PROHIBITED,
            TxnHardNegative.NEAR_SYNONYM_GL,
            TxnHardNegative.SPLIT_ALLOCATION,
            TxnHardNegative.REFUND,
            TxnHardNegative.PERSONAL_LOOKING_ALLOWED,
        )
        hard = _stable_choice(h >> 3, options)
    elif difficulty == Difficulty.MEDIUM:
        options = (
            TxnHardNegative.NEAR_SYNONYM_GL,
            TxnHardNegative.THRESHOLD_BOUNDARY,
            TxnHardNegative.MISLEADING_DESCRIPTOR,
            TxnHardNegative.NONE,
        )
        hard = _stable_choice(h >> 3, options)

    amount_minor = 1_000 + (h % 50_000) * 10
    if hard == TxnHardNegative.THRESHOLD_BOUNDARY:
        amount_minor = 5_001  # just over POL-MEAL-001
        meal_vendors = [v for v in vendors if v.category == "meals"]
        if meal_vendors:
            vendor = meal_vendors[0]
    if hard in {TxnHardNegative.CAPEX_OPEX, TxnHardNegative.CONFLICTING_RULES}:
        amount_minor = 5_000_000
        capex_vendors = [v for v in vendors if v.default_gl == "1500"]
        vendor = capex_vendors[0] if capex_vendors else vendor
    if hard == TxnHardNegative.REFUND:
        amount_minor = abs(amount_minor)

    descriptor = vendor.descriptors[h % len(vendor.descriptors)]
    if hard == TxnHardNegative.MISLEADING_DESCRIPTOR:
        descriptor = f"{descriptor} CONSULTING RETAINER"
    if hard == TxnHardNegative.PERSONAL_LOOKING_ALLOWED:
        descriptor = f"GIFT *{vendor.name.upper()} TEAM OFFSITE"
    if hard == TxnHardNegative.ALLOWED_LOOKING_PROHIBITED:
        descriptor = "AMZN GIFT CARD PERSONAL"

    is_refund = hard == TxnHardNegative.REFUND
    action, gl, rule_ids = resolve_policy(
        policies,
        descriptor=descriptor,
        amount_minor=amount_minor,
        category=vendor.category,
        hard_negative=hard,
    )

    if hard == TxnHardNegative.NEAR_SYNONYM_GL and gl == "6100":
        # Oracle keeps correct 6100; renderer will surface 6105 as distractor.
        pass

    split_lines: tuple[tuple[str, int], ...] = ()
    if hard == TxnHardNegative.SPLIT_ALLOCATION:
        a = amount_minor // 2
        b = amount_minor - a
        gl = "6600" if any(p.gl_account == "6600" for p in policies) else gl
        other = "6400" if gl != "6400" else "6500"
        split_lines = ((gl, a), (other, b))
        action = PolicyAction.REVIEW
        svc = next((p for p in policies if p.rule_id == "POL-SVC-006"), None)
        rule_ids = (svc.rule_id,) if svc is not None else rule_ids

    txn_id = f"txn_{entity_id}_{h % 10_000:04d}"
    month = (h % 12) + 1
    day = (h % 27) + 1
    return TransactionLatent(
        txn_id=txn_id,
        vendor_id=vendor.vendor_id,
        descriptor=descriptor,
        amount_minor=amount_minor,
        currency="USD",
        date=f"2026-{month:02d}-{day:02d}",
        entity_id=entity_id,
        cost_center=("ops", "eng", "finance", "sales")[h % 4],
        is_refund=is_refund,
        hard_negative=hard,
        gl_account=gl,
        policy_action=action,
        applied_rule_ids=rule_ids,
        split_lines=split_lines,
    )


def _build_variance(
    h: int,
    entity_id: str,
    difficulty: Difficulty,
    ood: bool,
) -> VarianceLatent:
    if difficulty == Difficulty.EASY:
        regime = VarianceRegime.SIMPLE
    elif difficulty == Difficulty.MEDIUM:
        regime = _stable_choice(
            h,
            (VarianceRegime.OFFSET, VarianceRegime.PRICE_VOLUME, VarianceRegime.SIMPLE),
        )
    else:
        regime = _stable_choice(
            h,
            (
                VarianceRegime.OFFSET,
                VarianceRegime.FX,
                VarianceRegime.TIE,
                VarianceRegime.HIDDEN_SUBTOTAL,
                VarianceRegime.PRICE_VOLUME,
            ),
        )
    if ood:
        regime = _stable_choice(
            h >> 2,
            (VarianceRegime.FX, VarianceRegime.OFFSET, VarianceRegime.PRICE_VOLUME),
        )

    budget = 1_000_000 + (h % 20) * 50_000
    if regime == VarianceRegime.SIMPLE:
        impact = 50_000 + (h % 10) * 10_000
        # Favorable: actual < budget for expense → positive profit impact
        actual = budget - impact
        drivers = (
            VarianceDriverLatent("headcount", impact, "budget_hc", "expense"),
        )
        other = 0
        rule = "VAR-MATERIAL-001"
    elif regime == VarianceRegime.TIE:
        drivers = (
            VarianceDriverLatent("alpha_cost", -100_000, "alpha_actual", "expense"),
            VarianceDriverLatent("beta_cost", -100_000, "beta_actual", "expense"),
        )
        other = 0
        actual = budget + 200_000
        rule = "VAR-TIEBREAK-001"
    elif regime == VarianceRegime.FX:
        drivers = (
            VarianceDriverLatent("volume", -350_000, "volume_actual", "volume"),
            VarianceDriverLatent("price", 200_000, "price_actual", "price"),
            VarianceDriverLatent("fx", -50_000, "fx_rate", "fx"),
        )
        other = -20_000
        actual = budget + 220_000
        rule = "VAR-FX-002"
    elif regime == VarianceRegime.PRICE_VOLUME:
        drivers = (
            VarianceDriverLatent("price", 80_000 + (h % 5) * 1_000, "price_actual", "price"),
            VarianceDriverLatent("volume", -(120_000 + (h % 5) * 1_000), "volume_actual", "volume"),
        )
        other = -10_000
        actual = budget + 50_000 + (h % 5) * 1_000
        rule = "VAR-MATERIAL-005"
    elif regime == VarianceRegime.HIDDEN_SUBTOTAL:
        drivers = (
            VarianceDriverLatent("cloud_usage", -300_000, "actual_cloud", "expense"),
            VarianceDriverLatent("support_volume", -90_000, "tickets_actual", "expense"),
        )
        other = -30_000
        actual = budget + 420_000
        rule = "VAR-MATERIAL-005"
    else:  # OFFSET
        drivers = (
            VarianceDriverLatent("cloud_usage", -300_000, "actual_cloud", "expense"),
            VarianceDriverLatent("support_volume", -90_000, "tickets_actual", "expense"),
        )
        other = -30_000
        actual = budget + 420_000
        rule = "VAR-MATERIAL-005"
        if ood:
            drivers = (
                VarianceDriverLatent("new_mix_a", -180_000, "mix_a_actual", "expense"),
                VarianceDriverLatent("new_mix_b", 90_000, "mix_b_actual", "revenue"),
                VarianceDriverLatent("new_mix_c", -40_000, "mix_c_actual", "fx"),
            )
            other = -15_000
            actual = budget + 145_000

    return VarianceLatent(
        period=f"2026-Q{(h % 4) + 1}",
        budget_minor=budget,
        actual_minor=actual,
        drivers=drivers,
        other_impact_minor=other,
        materiality_rule_id=rule,
        regime=regime,
        entity_id=entity_id,
    )


def _build_cash(
    h: int,
    entity_id: str,
    close_period: str,
    difficulty: Difficulty,
    ood: bool,
) -> CashLatent:
    if difficulty == Difficulty.EASY:
        regime = CashRegime.CLEAN_MATCH
    elif difficulty == Difficulty.MEDIUM:
        regime = _stable_choice(
            h,
            (CashRegime.BANK_FEE, CashRegime.DEPOSIT_IN_TRANSIT, CashRegime.CLEAN_MATCH),
        )
    else:
        regime = _stable_choice(
            h,
            (
                CashRegime.BANK_FEE,
                CashRegime.STALE_CHECK,
                CashRegime.DUPLICATE,
                CashRegime.ONE_TO_MANY,
                CashRegime.PARTIAL,
            ),
        )
    if ood:
        regime = _stable_choice(
            h >> 1,
            (CashRegime.ONE_TO_MANY, CashRegime.PARTIAL, CashRegime.DUPLICATE),
        )

    base = 8_000_000 + (h % 100) * 1_000
    b1 = BookEntry(f"b_{entity_id}_1", 250_000, "2026-03-01", "customer remittance")
    k1 = BankEvent(f"k_{entity_id}_1", 250_000, "2026-03-02", "ACH CREDIT", "clearing")

    book_entries = [b1]
    bank_events = [k1]
    matched: list[tuple[tuple[str, ...], tuple[str, ...]]] = [((b1.entry_id,), (k1.event_id,))]
    exceptions: list[tuple[str, tuple[str, ...], int]] = []
    book_balance = base
    bank_balance = base

    if regime == CashRegime.BANK_FEE:
        fee = BankEvent(f"k_{entity_id}_fee", -3_500, "2026-03-03", "SERVICE CHARGE", "fee")
        bank_events.append(fee)
        exceptions.append(("bank_fee", (fee.event_id,), 3_500))
        bank_balance = base - 3_500
        # adjusted bank adds back fee → equals book
    elif regime == CashRegime.DEPOSIT_IN_TRANSIT:
        b2 = BookEntry(f"b_{entity_id}_2", 50_000, "2026-03-04", "deposit in transit")
        book_entries.append(b2)
        exceptions.append(("deposit_in_transit", (b2.entry_id,), 50_000))
        book_balance = base + 50_000
        bank_balance = base
    elif regime == CashRegime.STALE_CHECK:
        b2 = BookEntry(f"b_{entity_id}_2", -12_000, "2026-01-15", "stale check")
        book_entries.append(b2)
        exceptions.append(("stale_check", (b2.entry_id,), 12_000))
        book_balance = base - 12_000
        bank_balance = base
    elif regime == CashRegime.DUPLICATE:
        dup = BankEvent(f"k_{entity_id}_dup", 250_000, "2026-03-02", "ACH CREDIT DUP", "duplicate")
        bank_events.append(dup)
        exceptions.append(("duplicate", (dup.event_id,), 250_000))
        bank_balance = base + 250_000
    elif regime == CashRegime.ONE_TO_MANY:
        b2 = BookEntry(f"b_{entity_id}_2", 100_000, "2026-03-05", "split receipt book")
        k2 = BankEvent(f"k_{entity_id}_2", 60_000, "2026-03-05", "partial clear a", "clearing")
        k3 = BankEvent(f"k_{entity_id}_3", 40_000, "2026-03-05", "partial clear b", "clearing")
        book_entries.append(b2)
        bank_events.extend([k2, k3])
        matched.append(((b2.entry_id,), (k2.event_id, k3.event_id)))
        book_balance = base + 100_000
        bank_balance = base + 100_000
    elif regime == CashRegime.PARTIAL:
        b2 = BookEntry(f"b_{entity_id}_2", 80_000, "2026-03-06", "invoice partial")
        k2 = BankEvent(f"k_{entity_id}_2", 50_000, "2026-03-06", "partial payment", "clearing")
        book_entries.append(b2)
        bank_events.append(k2)
        exceptions.append(("partial_settlement", (b2.entry_id, k2.event_id), 30_000))
        book_balance = base + 80_000
        bank_balance = base + 50_000

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


def _fnv(text: str) -> int:
    h = 2166136261
    for ch in text.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _mix(*parts: int) -> int:
    h = 0x9E3779B9
    for p in parts:
        h ^= (p + 0x9E3779B9 + ((h << 6) & 0xFFFFFFFF) + (h >> 2)) & 0xFFFFFFFF
        h &= 0xFFFFFFFF
    return h
