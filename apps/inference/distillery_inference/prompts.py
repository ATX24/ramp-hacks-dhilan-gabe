"""Deterministic prompt construction for finance Demo tasks."""

from __future__ import annotations

import json
from typing import Any

from distillery_inference.schemas import FinanceTaskId

SYSTEM_PREAMBLE = (
    "You are a finance analysis model. "
    "Return exactly one JSON object and no prose or markdown. "
    "Do not wrap the object in tasks, result, output, or an array. "
    "Use exactly the top-level keys shown in the schema shape example."
)

TASK_SCHEMA_EXAMPLES: dict[FinanceTaskId, dict[str, Any]] = {
    "transaction_review": {
        "task": "transaction_review",
        "schema_version": "transaction_review.v1",
        "gl_account": "example_account",
        "journal_entry": [
            {"account": "example_account", "amount_minor": 1, "side": "debit"},
            {"account": "example_offset", "amount_minor": 1, "side": "credit"},
        ],
        "policy_action": "review",
        "rule_ids": [],
        "evidence": [{"field": "example_field", "source_id": "example_source", "value": "example"}],
        "confidence": 0.5,
    },
    "variance_analysis": {
        "task": "variance_analysis",
        "schema_version": "variance_analysis.v1",
        "profit_impact_minor": 1,
        "direction": "favorable",
        "top_drivers": [{"driver_id": "example_driver", "impact_minor": 1, "rank": 1}],
        "other_impact_minor": 0,
        "rule_ids": [],
        "evidence_ids": [],
        "confidence": 0.5,
    },
    "cash_reconciliation": {
        "task": "cash_reconciliation",
        "schema_version": "cash_reconciliation.v1",
        "status": "balanced",
        "matched_groups": [
            {
                "book_ids": ["example_book_id"],
                "bank_ids": ["example_bank_id"],
            }
        ],
        "exceptions": [],
        "adjusted_book_balance_minor": 1,
        "adjusted_bank_balance_minor": 1,
        "difference_minor": 0,
        "confidence": 0.5,
    },
}

TASK_INVARIANTS: dict[FinanceTaskId, str] = {
    "transaction_review": (
        "journal_entry must contain positive debit and credit lines whose amounts balance"
    ),
    "variance_analysis": (
        "direction must be favorable or unfavorable; top_drivers must contain objects "
        "with driver_id, impact_minor, and contiguous rank; driver impacts plus "
        "other_impact_minor must equal nonzero profit_impact_minor"
    ),
    "cash_reconciliation": (
        "difference_minor must equal adjusted_book_balance_minor minus "
        "adjusted_bank_balance_minor; status must be balanced or exceptions; each "
        "matched_groups item must be an object containing nonempty book_ids and bank_ids "
        "arrays copied from the input; balanced requires equal adjusted balances, zero "
        "difference, and no exceptions; always use balanced when those three conditions "
        "hold; use exceptions only with a nonempty exceptions array"
    ),
}


def _cash_arithmetic_hint(example_input: dict[str, Any]) -> str:
    book_balance = example_input.get("book_balance_minor")
    bank_balance = example_input.get("bank_balance_minor")
    if (
        not isinstance(book_balance, int)
        or isinstance(book_balance, bool)
        or not isinstance(bank_balance, int)
        or isinstance(bank_balance, bool)
    ):
        return ""
    raw_difference = book_balance - bank_balance
    hint = (
        "Verified input arithmetic: raw book minus bank balance is "
        f"{book_balance} - {bank_balance} = {raw_difference}."
    )
    book_entries = example_input.get("book_entries")
    bank_events = example_input.get("bank_events")
    if not isinstance(book_entries, list) or not isinstance(bank_events, list):
        return hint
    book_amounts = [
        entry.get("amount_minor")
        for entry in book_entries
        if isinstance(entry, dict) and isinstance(entry.get("amount_minor"), int)
    ]
    bank_amounts = [
        event.get("amount_minor")
        for event in bank_events
        if isinstance(event, dict) and isinstance(event.get("amount_minor"), int)
    ]
    if (
        len(book_amounts) == len(book_entries)
        and len(bank_amounts) == len(bank_events)
        and sorted(book_amounts) == sorted(bank_amounts)
        and raw_difference == 0
    ):
        hint += (
            " The supplied entry amounts match exactly and the balances are equal, "
            "so no exception is evidenced. Required values for this exact-match case: "
            f"adjusted_book_balance_minor={book_balance}, "
            f"adjusted_bank_balance_minor={bank_balance}, difference_minor=0, "
            'status="balanced", exceptions=[].'
        )
    return hint


def _task_arithmetic_hint(
    task: FinanceTaskId,
    example_input: dict[str, Any],
) -> str:
    if task == "cash_reconciliation":
        return _cash_arithmetic_hint(example_input)
    return ""


def build_messages(
    *,
    task: FinanceTaskId,
    example_input: dict[str, Any],
) -> list[dict[str, str]]:
    schema_example = json.dumps(TASK_SCHEMA_EXAMPLES[task], sort_keys=True)
    input_payload = json.dumps(example_input, sort_keys=True)
    arithmetic_hint = _task_arithmetic_hint(task, example_input)
    user_content = (
        "Schema shape example only; do not reuse its placeholder values:\n"
        f"{schema_example}\n"
        "Analyze this input and fill every schema key from the evidence. "
        f'The task value must be "{task}" and the schema_version must be '
        f'"{task}.v1". Before returning, verify every numeric invariant exactly. '
        "Copy evidence identifiers exactly; never emit a bare string where the example "
        "shows an object or array.\n"
        f"{arithmetic_hint}\n"
        f"Input:\n{input_payload}"
    )
    return [
        {
            "role": "system",
            "content": f"{SYSTEM_PREAMBLE} {TASK_INVARIANTS[task]}.",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def render_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Fallback plain prompt when a chat template is unavailable."""
    parts: list[str] = []
    for message in messages:
        parts.append(f"{message['role'].upper()}: {message['content']}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)
