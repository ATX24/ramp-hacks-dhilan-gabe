"""Model-facing renderers over hidden finance-world state."""

from __future__ import annotations

import hashlib
from typing import Any

from distillery.contracts.tasks import Difficulty, TaskId
from distillery.data.world import (
    IID_TEMPLATE_FAMILIES,
    OOD_TEMPLATE_FAMILIES,
    LatentWorld,
    VarianceObservation,
)


def select_template_family(
    task: TaskId,
    *,
    difficulty: Difficulty,
    ood: bool,
    index: int,
    family_key: str = "",
) -> str:
    """Select a base family and make its provenance identity group-specific."""
    families = OOD_TEMPLATE_FAMILIES[task] if ood else IID_TEMPLATE_FAMILIES[task]
    bump = {"easy": 0, "medium": 1, "hard": 2}[difficulty.value]
    base = families[(index + bump) % len(families)]
    key = family_key or f"{task.value}|{difficulty.value}|{index}|{int(ood)}"
    suffix = hashlib.sha256(f"{key}|{base}".encode()).hexdigest()[:10]
    return f"{base}__{suffix}"


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


def render_transaction_review(
    world: LatentWorld,
    *,
    template_family: str,
) -> dict[str, Any]:
    transaction = world.transaction
    if transaction is None:
        raise ValueError("missing transaction")
    vendor = next(vendor for vendor in world.vendors if vendor.vendor_id == transaction.vendor_id)
    prompt = {
        "txn_card_packet": "Review the card transaction using the supplied controls.",
        "txn_policy_memo": "Code the transaction and apply policy precedence.",
        "txn_ledger_excerpt": "Prepare a balanced journal and policy disposition.",
        "txn_receipt_bundle": "Review the receipt context and accounting treatment.",
        "txn_matrix_packet": "Resolve the transaction against the control matrix.",
        "txn_account_crosswalk": "Use the account crosswalk and policy schedule.",
        "txn_control_note": "Determine coding and disposition from the control note.",
    }.get(_template_base(template_family), "Review the transaction.")

    policy_rules = [
        {
            "rule_id": rule.rule_id,
            "precedence": rule.precedence,
            "action": rule.action.value,
            "gl_account": rule.gl_account,
            "min_amount_minor": rule.min_amount_minor,
            "max_amount_minor": rule.max_amount_minor,
            "keywords": list(rule.keywords),
            "category": rule.category,
            "vendor_ids": list(rule.vendor_ids),
            "text": rule.text,
        }
        for rule in world.policies
    ]
    return {
        "prompt": prompt,
        "txn_id": transaction.txn_id,
        "amount_minor": transaction.amount_minor,
        "currency": transaction.currency,
        "date": transaction.date,
        "descriptor": transaction.descriptor,
        "vendor_id": vendor.vendor_id,
        "vendor": vendor.name,
        "expense_category": vendor.category,
        "entity_id": transaction.entity_id,
        "cost_center": transaction.cost_center,
        "chart_of_accounts": [
            {
                "code": account.code,
                "name": account.name,
                "type": account.account_type,
            }
            for account in world.chart_of_accounts
        ],
        "policy_excerpt": " ".join(rule.text for rule in world.policies),
        "policy_rules": policy_rules,
    }


def render_variance_analysis(
    world: LatentWorld,
    *,
    template_family: str,
) -> dict[str, Any]:
    variance = world.variance
    if variance is None:
        raise ValueError("missing variance")
    prompt = {
        "var_operating_table": "Analyze the operating variance using exact minor units.",
        "var_driver_packet": "Rank the drivers that explain the profit variance.",
        "var_close_memo": "Prepare the period-close variance bridge.",
        "var_dimension_slice": "Analyze the supplied dimensional P&L slice.",
        "var_bridge_packet": "Build a driver bridge from the supplied comparisons.",
        "var_unit_economics": "Decompose the unit economics comparison.",
        "var_currency_schedule": "Analyze the operating and currency schedule.",
    }.get(_template_base(template_family), "Analyze the variance.")
    result: dict[str, Any] = {
        "prompt": prompt,
        "period": variance.period,
        "entity_id": variance.entity_id,
        "budget_minor": variance.budget_minor,
        "actual_minor": variance.actual_minor,
        "pnl_basis": "profit impact equals budget net expense minus actual net expense",
        "driver_observations": [
            _render_variance_observation(observation) for observation in variance.drivers
        ],
        "analysis_rules": [
            {"rule_id": rule_id, "text": text} for rule_id, text in variance.analysis_rules
        ],
    }
    if variance.unallocated:
        result["unallocated_line_items"] = [
            _render_variance_observation(observation) for observation in variance.unallocated
        ]
    return result


def render_cash_reconciliation(
    world: LatentWorld,
    *,
    template_family: str,
) -> dict[str, Any]:
    cash = world.cash
    if cash is None:
        raise ValueError("missing cash")
    prompt = {
        "cash_statement_packet": "Reconcile the ledger to the bank statement.",
        "cash_close_table": "Identify reconciling items and adjusted balances.",
        "cash_ledger_extract": "Match the ledger extract to statement events.",
        "cash_aggregation_sheet": "Resolve grouped settlements and exceptions.",
        "cash_collision_worksheet": "Resolve ambiguous same-amount statement events.",
        "cash_settlement_packet": "Reconcile partial and aggregated settlements.",
    }.get(_template_base(template_family), "Reconcile cash.")
    return {
        "prompt": prompt,
        "entity_id": cash.entity_id,
        "close_period": cash.close_period,
        "book_balance_minor": cash.book_balance_minor,
        "bank_balance_minor": cash.bank_balance_minor,
        "book_entries": [
            {
                "id": entry.entry_id,
                "amount_minor": entry.amount_minor,
                "date": entry.date,
                "memo": entry.memo,
            }
            for entry in cash.book_entries
        ],
        "bank_events": [
            {
                "id": event.event_id,
                "amount_minor": event.amount_minor,
                "date": event.date,
                "memo": event.memo,
                "type": event.event_type,
            }
            for event in cash.bank_events
        ],
    }


def _render_variance_observation(
    observation: VarianceObservation,
) -> dict[str, Any]:
    return {
        "source_id": observation.source_id,
        "driver_id": observation.driver_id,
        "pnl_type": observation.pnl_type,
        "budget_minor": observation.budget_minor,
        "actual_minor": observation.actual_minor,
        "kind": observation.kind,
    }


def _template_base(template_family: str) -> str:
    return template_family.split("__", maxsplit=1)[0]
