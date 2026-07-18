"""Natural-language / structured input renderers over latent world state."""

from __future__ import annotations

from typing import Any

from distillery.contracts.tasks import Difficulty, TaskId
from distillery.data.world import (
    IID_TEMPLATE_FAMILIES,
    OOD_TEMPLATE_FAMILIES,
    LatentWorld,
    TxnHardNegative,
)


def select_template_family(
    task: TaskId,
    *,
    difficulty: Difficulty,
    ood: bool,
    index: int,
) -> str:
    families = OOD_TEMPLATE_FAMILIES[task] if ood else IID_TEMPLATE_FAMILIES[task]
    # Difficulty nudges family choice without leaking labels.
    bump = {"easy": 0, "medium": 1, "hard": 2}[difficulty.value]
    return families[(index + bump) % len(families)]


def render_input(
    world: LatentWorld,
    task: TaskId,
    *,
    template_family: str,
) -> dict[str, Any]:
    if task == TaskId.TRANSACTION_REVIEW:
        return render_transaction_review(world, template_family=template_family)
    if task == TaskId.VARIANCE_ANALYSIS:
        return render_variance_analysis(world, template_family=template_family)
    if task == TaskId.CASH_RECONCILIATION:
        return render_cash_reconciliation(world, template_family=template_family)
    raise ValueError(f"no renderer for task {task}")


def render_transaction_review(world: LatentWorld, *, template_family: str) -> dict[str, Any]:
    txn = world.transaction
    if txn is None:
        raise ValueError("missing transaction")

    vendor = next(v for v in world.vendors if v.vendor_id == txn.vendor_id)
    gl_codes = sorted({a.code for a in world.chart_of_accounts})
    # Near-synonym hard negative: surface distractor GL.
    candidates = [txn.gl_account, "2100"]
    if txn.hard_negative == TxnHardNegative.NEAR_SYNONYM_GL:
        candidates.append("6105" if txn.gl_account == "6100" else "6100")
    if txn.hard_negative == TxnHardNegative.CAPEX_OPEX:
        candidates.extend(["1500", "6400"])
    candidates = list(dict.fromkeys(c for c in candidates if c in gl_codes or True))

    policy_lines = []
    for rule in sorted(world.policies, key=lambda r: r.precedence)[:6]:
        cap = (
            f"max=${rule.max_amount_minor / 100:.2f}"
            if rule.max_amount_minor is not None
            else "no max"
        )
        policy_lines.append(
            f"{rule.rule_id} (prec={rule.precedence}): "
            f"{rule.action.value} -> {rule.gl_account} ({cap})"
        )

    prompt_prefix = {
        "txn_meal_v1": "Review this meal/card spend against policy.",
        "txn_policy_v3": "Apply the versioned spend policy to code this transaction.",
        "txn_capex_v2": "Decide whether this purchase is capex or opex and the policy action.",
        "txn_saas_v1": "Code this SaaS/software charge.",
        "txn_conflict_v1": "Resolve conflicting policy rules using explicit precedence.",
        "txn_ood_policy_mix_v1": "OOD policy mix: apply held-out rule combinations.",
        "txn_ood_gl_desc_v1": "OOD GL/description pairing; do not rely on memorized templates.",
        "txn_ood_threshold_v1": "OOD threshold boundary case.",
    }.get(template_family, "Review the transaction.")

    return {
        "prompt": prompt_prefix,
        "template_family": template_family,
        "txn_id": txn.txn_id,
        "amount_minor": txn.amount_minor,
        "currency": txn.currency,
        "date": txn.date,
        "descriptor": txn.descriptor,
        "vendor": vendor.name,
        "entity_id": txn.entity_id,
        "cost_center": txn.cost_center,
        "gl_candidates": candidates,
        "chart_of_accounts": [
            {"code": a.code, "name": a.name, "type": a.account_type}
            for a in world.chart_of_accounts
        ],
        "policy_excerpt": " | ".join(policy_lines),
        "policy_rules": [
            {
                "rule_id": r.rule_id,
                "precedence": r.precedence,
                "action": r.action.value,
                "gl_account": r.gl_account,
                "max_amount_minor": r.max_amount_minor,
            }
            for r in world.policies
        ],
        "is_refund_hint": txn.is_refund,
        "hard_negative": txn.hard_negative.value,
    }


def render_variance_analysis(world: LatentWorld, *, template_family: str) -> dict[str, Any]:
    var = world.variance
    if var is None:
        raise ValueError("missing variance")

    prompt_prefix = {
        "var_simple_v1": "Explain the P&L variance versus budget.",
        "var_drivers_v2": "Rank the material drivers of profit impact.",
        "var_offset_v1": "Offsetting drivers are present; close the arithmetic.",
        "var_tie_v1": "Break ties by driver_id ascending after |impact|.",
        "var_ood_driver_mix_v1": "OOD driver mixture with novel sign interactions.",
        "var_ood_sign_v1": "OOD sign-convention stress case.",
        "var_ood_fx_v1": "OOD FX + price/volume decomposition.",
    }.get(template_family, "Analyze the variance.")

    drivers = [
        {
            "driver_id": d.driver_id,
            "impact_hint_minor": d.impact_minor,
            "evidence_id": d.evidence_id,
            "kind": d.kind,
        }
        for d in var.drivers
    ]

    return {
        "prompt": prompt_prefix,
        "template_family": template_family,
        "period": var.period,
        "entity_id": var.entity_id,
        "budget_minor": var.budget_minor,
        "actual_minor": var.actual_minor,
        "drivers": drivers,
        "materiality_rule_id": var.materiality_rule_id,
        "regime": var.regime.value,
        "sign_convention": "positive profit_impact is favorable",
        "other_bucket_present": var.other_impact_minor != 0,
    }


def render_cash_reconciliation(world: LatentWorld, *, template_family: str) -> dict[str, Any]:
    cash = world.cash
    if cash is None:
        raise ValueError("missing cash")

    prompt_prefix = {
        "cash_match_v1": "Reconcile book ledger to bank statement.",
        "cash_exceptions_v1": "Identify fees, timing items, and exceptions.",
        "cash_partial_v1": "Handle partial settlements and many-to-one clears.",
        "cash_ood_agg_v1": "OOD aggregation pattern reconciliation.",
        "cash_ood_collision_v1": "OOD same-amount/date collision stress.",
    }.get(template_family, "Reconcile cash.")

    return {
        "prompt": prompt_prefix,
        "template_family": template_family,
        "entity_id": cash.entity_id,
        "close_period": cash.close_period,
        "book_balance_minor": cash.book_balance_minor,
        "bank_balance_minor": cash.bank_balance_minor,
        "book_entries": [
            {
                "id": e.entry_id,
                "amount_minor": e.amount_minor,
                "date": e.date,
                "memo": e.memo,
            }
            for e in cash.book_entries
        ],
        "bank_events": [
            {
                "id": e.event_id,
                "amount_minor": e.amount_minor,
                "date": e.date,
                "memo": e.memo,
                "type": e.event_type,
            }
            for e in cash.bank_events
        ],
        "regime": cash.regime.value,
    }
