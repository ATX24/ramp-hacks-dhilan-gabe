"""Teacher synthesis via programmatic Claude Opus usage.

The teacher (claude-opus-4-8) fills ONLY missing/rejected training/validation
responses. It never sees test prompts or expected outputs. Every generated
response records full provenance (model id, params) and passes deterministic
validation before acceptance."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..contracts.dataset import Example, canonical_json
from ..contracts.errors import TeacherUnavailable, OutputUseNotAllowed
from ..data.validate import validate_output

TEACHER_MODEL = "claude-opus-4-8"
TEACHER_PARAMS = {"max_tokens": 1024, "temperature": 0.0}

SYSTEM_PROMPT = (
    "You are a meticulous corporate finance engine. Respond with ONLY a single JSON object "
    "matching the requested task schema. No prose, no markdown fences. Amounts are integer "
    "minor units. Journal entries must balance exactly. Follow policy rule precedence."
)

TASK_INSTRUCTIONS = {
    "transaction_review": (
        'Task transaction_review: given the transaction, chart of accounts, and policies, return '
        '{"schema_version":"transaction_review.v1","task":"transaction_review","gl_account":...,'
        '"journal_entry":[{"account":...,"side":"debit|credit","amount_minor":int},...],'
        '"policy_action":"approve|review|reject","rule_ids":[...],"evidence":[{"source_id":...,"field":...,"value":...}],'
        '"confidence":float}. Debit the expense/asset account, credit the payment account.'
    ),
    "variance_analysis": (
        'Task variance_analysis: impact_minor for each driver is budget_minor - actual_minor for expenses '
        '(spend over budget is negative profit impact). Return '
        '{"schema_version":"variance_analysis.v1","task":"variance_analysis","profit_impact_minor":int,'
        '"direction":"favorable|unfavorable","top_drivers":[{"driver_id":...,"impact_minor":int,"rank":int}],'
        '"other_impact_minor":int,"rule_ids":["VAR-MATERIAL-005"],"evidence_ids":[...],"confidence":float}. '
        'top_drivers sorted by descending absolute impact, driver_id tiebreak; '
        'sum(top_drivers impacts) + other_impact_minor must equal profit_impact_minor.'
    ),
    "cash_reconciliation": (
        'Task cash_reconciliation: match book entries to bank events by amount; classify fees, duplicates, '
        'deposits in transit as exceptions. Return '
        '{"schema_version":"cash_reconciliation.v1","task":"cash_reconciliation","status":"clean|exceptions",'
        '"matched_groups":[{"book_ids":[...],"bank_ids":[...]}],"exceptions":[{"type":...,"event_ids":[...],'
        '"amount_minor":int}],"adjusted_book_balance_minor":int,"adjusted_bank_balance_minor":int,'
        '"difference_minor":int,"confidence":float}.'
    ),
}


def build_prompt(ex: Example) -> str:
    return TASK_INSTRUCTIONS[ex.task] + "\n\nINPUT:\n" + canonical_json(ex.input)


def _client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise TeacherUnavailable("ANTHROPIC_API_KEY is not set; teacher synthesis unavailable.")
    import anthropic
    return anthropic.Anthropic()


def synthesize_missing(examples: list[Example], out_path: Path, max_cost_usd: float = 25.0,
                       dry_run: bool = False) -> dict:
    """Fill missing/invalid responses for train/validation examples only."""
    for ex in examples:
        if ex.provenance.split in ("test_iid", "test_ood"):
            raise OutputUseNotAllowed(
                "Teacher must never see test prompts before final evaluation.",
                details={"example_id": ex.example_id})

    todo = [ex for ex in examples if not ex.response]
    stats = {"total": len(examples), "already_present": len(examples) - len(todo),
             "generated": 0, "rejected": 0, "teacher_model": TEACHER_MODEL,
             "teacher_params": TEACHER_PARAMS, "estimated_calls": len(todo)}
    if dry_run:
        return stats

    client = _client()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        for ex in todo:
            resp = client.messages.create(
                model=TEACHER_MODEL, system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_prompt(ex)}],
                **TEACHER_PARAMS,
            )
            text = resp.content[0].text
            obj, errs = validate_output(ex.task, text, ex.input)
            record = {
                "example_id": ex.example_id, "task": ex.task,
                "response": text, "validation_errors": errs,
                "accepted": not errs,
                "label_source": "teacher", "teacher_model": TEACHER_MODEL,
                "teacher_params": TEACHER_PARAMS,
                "usage": {"input_tokens": resp.usage.input_tokens,
                          "output_tokens": resp.usage.output_tokens},
            }
            f.write(json.dumps(record) + "\n")
            if errs:
                stats["rejected"] += 1
            else:
                ex.response = text
                ex.provenance.label_source = "teacher"
                ex.provenance.teacher_model = TEACHER_MODEL
                ex.provenance.teacher_params = TEACHER_PARAMS
                stats["generated"] += 1
    return stats


def materialize_oracle_responses(examples: list[Example]) -> int:
    """oracle_sft arm: use latent-oracle expected outputs as gold responses."""
    n = 0
    for ex in examples:
        if not ex.response:
            ex.response = canonical_json(ex.expected_output)
            ex.provenance.label_source = "oracle"
            n += 1
    return n
