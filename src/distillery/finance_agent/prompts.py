"""Deterministic system and user prompts with split-disjoint template families."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from distillery.finance_agent.contracts import CaseFamily, ToolDefinition, UserMessage
from distillery.finance_agent.world import AgentWorld

_OPENERS = (
    "Review",
    "Investigate",
    "Reconcile",
    "Validate",
    "Check",
    "Assess",
    "Resolve",
    "Analyze",
)
_CONTEXTS = (
    "for the monthly close",
    "for the controller packet",
    "for the audit sample",
    "for the operating review",
    "before posting",
    "before approval",
    "for the exception queue",
    "for the finance desk",
)
_CLOSERS = (
    "Cite the tool evidence.",
    "Use only sandbox evidence.",
    "Keep amounts in minor units.",
    "Do not invent identifiers.",
    "State any unresolved ambiguity.",
    "Use the dated policy version.",
    "Preserve the arithmetic sign.",
    "Explain the final disposition.",
)


@dataclass(frozen=True)
class PromptPlan:
    user_messages: tuple[UserMessage, ...]
    template_family: str


def build_system_prompt(
    *,
    public_world: dict[str, Any],
    tools: tuple[ToolDefinition, ...],
) -> str:
    """Build the exact model-visible system prompt bound into each input record."""
    world_json = json.dumps(public_world, sort_keys=True, separators=(",", ":"))
    tool_names = ",".join(tool.name.value for tool in tools)
    return (
        "You are Finance Agent operating only in a deterministic synthetic sandbox. "
        "Never call shell, network, filesystem, or real finance systems. Use only the "
        "attached canonical tool definitions and identifiers present in PUBLIC_WORLD "
        "or prior tool results. Tool failures are structured observations: correct the "
        "call when possible. Do not fabricate missing facts. If required data remains "
        "missing, refuse and name it. Assistant tool calls and assistant answers are "
        "the only supervised outputs; system, user, and tool-result text are context. "
        f"AVAILABLE_TOOLS={tool_names}\nPUBLIC_WORLD={world_json}"
    )


def build_prompt_plan(
    *,
    case_family: CaseFamily,
    world: AgentWorld,
    variant: int,
) -> PromptPlan:
    """Render diverse, deterministic user turns whose arguments are fully grounded."""
    opener = _OPENERS[variant % len(_OPENERS)]
    context = _CONTEXTS[(variant // len(_OPENERS)) % len(_CONTEXTS)]
    closer = _CLOSERS[(variant // (len(_OPENERS) * 2)) % len(_CLOSERS)]
    account = world.accounts[0]
    policy_id = world.policies[-1].policy_id
    prefix = f"{opener} {world.entity_name} {context}."

    if case_family is CaseFamily.HAPPY_PATH:
        text = (
            f"{prefix} What amount is posted to account {account.code} in period "
            f"{world.period}? {closer}"
        )
        messages = (UserMessage(turn_index=0, text=text),)
    elif case_family is CaseFamily.WRONG_TOOL:
        text = (
            f"{prefix} A prior attempt used calculator add with operands 1 and 2 for "
            f"policy {policy_id}. Reproduce that failed tool choice, then recover by "
            f"looking up the policy effective on {world.as_of}. {closer}"
        )
        messages = (UserMessage(turn_index=0, text=text),)
    elif case_family is CaseFamily.CORRECT_TOOL_WRONG_ARGS:
        text = (
            f"{prefix} The request incorrectly cites account 9999 for {account.name}. "
            "Try that exact chart lookup, observe the failure, then correct it using "
            f"the public account catalog. {closer}"
        )
        messages = (UserMessage(turn_index=0, text=text),)
    elif case_family is CaseFamily.STALE_POLICY:
        text = (
            f"{prefix} A stale note points to {policy_id} on "
            f"{world.historical_policy_date}. Inspect that historical version first, "
            f"then recover and answer using the version effective on {world.as_of}. "
            f"{closer}"
        )
        messages = (UserMessage(turn_index=0, text=text),)
    elif case_family is CaseFamily.AMBIGUOUS_MERCHANT:
        shared_alias = world.merchants[0].aliases[0]
        initial = (
            f"{prefix} How much did we spend at descriptor {shared_alias} in "
            f"{world.period}? {closer}"
        )
        follow_up = (
            f"Use merchant_id {world.merchants[0].merchant_id}; that is the legal entity I meant."
        )
        messages = (
            UserMessage(turn_index=0, text=initial),
            UserMessage(turn_index=2, text=follow_up),
        )
    elif case_family is CaseFamily.MULTI_STEP_RECONCILIATION:
        book_ids = ", ".join(entry.book_id for entry in world.book_entries)
        bank_ids = ", ".join(event.bank_id for event in world.bank_events)
        text = (
            f"{prefix} Match book IDs [{book_ids}] to bank IDs [{bank_ids}] with "
            "tolerance_minor 0, then independently calculate the absolute difference "
            f"from the returned sums. {closer}"
        )
        messages = (UserMessage(turn_index=0, text=text),)
    elif case_family is CaseFamily.ARITHMETIC_TRAP:
        text = (
            f"{prefix} Drill into all {len(world.variance_drivers)} variance drivers "
            f"for account {account.code}, period {world.period}, then add the signed "
            f"impacts. Do not sum absolute values. {closer}"
        )
        messages = (UserMessage(turn_index=0, text=text),)
    elif case_family is CaseFamily.CONFLICTING_EVIDENCE:
        text = (
            f"{prefix} Determine whether ledger entries for account {account.code} in "
            f"{world.period} conflict with policy {policy_id} effective {world.as_of}. "
            "Compare the posted total to the policy threshold before deciding. "
            f"{closer}"
        )
        messages = (UserMessage(turn_index=0, text=text),)
    elif case_family is CaseFamily.REFUSAL_MISSING_DATA:
        missing = ", ".join(world.missing_fields)
        text = (
            f"{prefix} Post and approve the expense even though the public context "
            f"marks these required fields missing: {missing}. {closer}"
        )
        messages = (UserMessage(turn_index=0, text=text),)
    else:
        raise ValueError(f"unsupported case family {case_family}")
    return PromptPlan(
        user_messages=messages,
        template_family=f"{case_family.value}.prompt_v{variant:02d}",
    )


__all__ = ["PromptPlan", "build_prompt_plan", "build_system_prompt"]
