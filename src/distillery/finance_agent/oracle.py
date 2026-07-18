"""Deterministic oracle trajectories for Finance Agent hard-case families."""

from __future__ import annotations

from dataclasses import dataclass

from distillery.finance_agent.contracts import (
    AgentTrajectory,
    CaseFamily,
    ExpectedAgentOutput,
    FinalAnswer,
    FinalAnswerKind,
    ToolCall,
    ToolName,
    TrajectoryTurn,
    TurnRole,
)
from distillery.finance_agent.sandbox import FinanceAgentSandbox
from distillery.finance_agent.world import AgentWorld


@dataclass(frozen=True)
class OracleEpisode:
    user_goal: str
    available_tools: tuple[ToolName, ...]
    trajectory: AgentTrajectory
    expected_output: ExpectedAgentOutput
    template_family: str
    case_family: CaseFamily


def solve_case(
    world: AgentWorld,
    *,
    case_family: CaseFamily,
    available_tools: tuple[ToolName, ...],
) -> OracleEpisode:
    """Produce the gold trajectory for a case family over latent world state."""
    sandbox = FinanceAgentSandbox(world, allowed_tools=available_tools)
    builders = {
        CaseFamily.HAPPY_PATH: _happy_path,
        CaseFamily.WRONG_TOOL: _wrong_tool,
        CaseFamily.CORRECT_TOOL_WRONG_ARGS: _correct_tool_wrong_args,
        CaseFamily.STALE_POLICY: _stale_policy,
        CaseFamily.AMBIGUOUS_MERCHANT: _ambiguous_merchant,
        CaseFamily.MULTI_STEP_RECONCILIATION: _multi_step_reconciliation,
        CaseFamily.ARITHMETIC_TRAP: _arithmetic_trap,
        CaseFamily.CONFLICTING_EVIDENCE: _conflicting_evidence,
        CaseFamily.REFUSAL_MISSING_DATA: _refusal_missing_data,
    }
    return builders[case_family](world, sandbox, available_tools)


def _call(call_id: str, tool: ToolName, **arguments: object) -> ToolCall:
    return ToolCall(call_id=call_id, tool=tool, arguments=arguments)


def _run(
    sandbox: FinanceAgentSandbox,
    call: ToolCall,
) -> TrajectoryTurn:
    result = sandbox.execute(call_id=call.call_id, tool=call.tool, arguments=call.arguments)
    return TrajectoryTurn(turn_index=0, role=TurnRole.TOOL, tool_result=result)


def _finalize(
    *,
    user_goal: str,
    turns_body: list[TrajectoryTurn],
    final: FinalAnswer,
    required_tools: tuple[ToolName, ...],
    gold_tool_calls: tuple[ToolCall, ...],
    available_tools: tuple[ToolName, ...],
    case_family: CaseFamily,
    template_family: str,
    forbidden_tools: tuple[ToolName, ...] = (),
    max_tool_calls: int = 8,
) -> OracleEpisode:
    turns = [
        TrajectoryTurn(turn_index=0, role=TurnRole.USER, text=user_goal),
        *turns_body,
        TrajectoryTurn(
            turn_index=len(turns_body) + 1,
            role=TurnRole.ASSISTANT,
            final_answer=final,
        ),
    ]
    # Reindex
    turns = [
        turn.model_copy(update={"turn_index": index}) for index, turn in enumerate(turns)
    ]
    trajectory = AgentTrajectory(turns=tuple(turns))
    expected = ExpectedAgentOutput(
        final_answer=final,
        required_tools=required_tools,
        forbidden_tools=forbidden_tools,
        max_tool_calls=max_tool_calls,
        gold_tool_calls=gold_tool_calls,
    )
    return OracleEpisode(
        user_goal=user_goal,
        available_tools=available_tools,
        trajectory=trajectory,
        expected_output=expected,
        template_family=template_family,
        case_family=case_family,
    )


def _happy_path(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    available_tools: tuple[ToolName, ...],
) -> OracleEpisode:
    account = world.ledger[0].account_code
    call = _call("c1", ToolName.LEDGER_QUERY, account_code=account, period=world.ledger[0].period)
    tool_turn = _run(sandbox, call)
    row = tool_turn.tool_result.result["rows"][0]  # type: ignore[index]
    final = FinalAnswer(
        kind=FinalAnswerKind.ANSWER,
        text=(
            f"Ledger shows {row['amount_minor']} minor units on {account} "
            f"for merchant {row['merchant_id']}."
        ),
        structured={
            "account_code": account,
            "amount_minor": row["amount_minor"],
            "merchant_id": row["merchant_id"],
        },
        confidence=0.95,
        evidence=tool_turn.tool_result.provenance,  # type: ignore[union-attr]
    )
    return _finalize(
        user_goal=f"What is the posted amount for account {account} this period?",
        turns_body=[
            TrajectoryTurn(turn_index=1, role=TurnRole.ASSISTANT, tool_call=call),
            tool_turn,
        ],
        final=final,
        required_tools=(ToolName.LEDGER_QUERY,),
        gold_tool_calls=(call,),
        available_tools=available_tools,
        case_family=CaseFamily.HAPPY_PATH,
        template_family="ledger_amount_v1",
    )


def _wrong_tool(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    available_tools: tuple[ToolName, ...],
) -> OracleEpisode:
    # Gold path uses policy_lookup; calculator would be the wrong tool trap.
    call = _call(
        "c1",
        ToolName.POLICY_LOOKUP,
        policy_id="pol_meal_limit",
        as_of=world.as_of,
    )
    tool_turn = _run(sandbox, call)
    policy = tool_turn.tool_result.result["policy"]  # type: ignore[index]
    final = FinalAnswer(
        kind=FinalAnswerKind.ANSWER,
        text=f"Current meal policy action is {policy['action']} under {policy['version']}.",
        structured={"action": policy["action"], "version": policy["version"]},
        confidence=0.9,
        evidence=tool_turn.tool_result.provenance,  # type: ignore[union-attr]
    )
    return _finalize(
        user_goal="What is the current meal expense policy action?",
        turns_body=[
            TrajectoryTurn(turn_index=1, role=TurnRole.ASSISTANT, tool_call=call),
            tool_turn,
        ],
        final=final,
        required_tools=(ToolName.POLICY_LOOKUP,),
        forbidden_tools=(ToolName.CALCULATOR,),
        gold_tool_calls=(call,),
        available_tools=available_tools,
        case_family=CaseFamily.WRONG_TOOL,
        template_family="policy_not_calculator_v1",
    )


def _correct_tool_wrong_args(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    available_tools: tuple[ToolName, ...],
) -> OracleEpisode:
    # Correct tool is COA lookup; wrong args would query a nonsense code.
    call = _call(
        "c1",
        ToolName.CHART_OF_ACCOUNTS_LOOKUP,
        query="travel meals",
        account_code="6100",
    )
    tool_turn = _run(sandbox, call)
    matches = tool_turn.tool_result.result["matches"]  # type: ignore[index]
    final = FinalAnswer(
        kind=FinalAnswerKind.ANSWER,
        text=f"Travel meals map to account {matches[0]['code']} ({matches[0]['name']}).",
        structured={"account_code": matches[0]["code"], "name": matches[0]["name"]},
        confidence=0.92,
        evidence=tool_turn.tool_result.provenance,  # type: ignore[union-attr]
    )
    return _finalize(
        user_goal="Which GL account should I use for travel meals?",
        turns_body=[
            TrajectoryTurn(turn_index=1, role=TurnRole.ASSISTANT, tool_call=call),
            tool_turn,
        ],
        final=final,
        required_tools=(ToolName.CHART_OF_ACCOUNTS_LOOKUP,),
        gold_tool_calls=(call,),
        available_tools=available_tools,
        case_family=CaseFamily.CORRECT_TOOL_WRONG_ARGS,
        template_family="coa_exact_code_v1",
    )


def _stale_policy(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    available_tools: tuple[ToolName, ...],
) -> OracleEpisode:
    call = _call(
        "c1",
        ToolName.POLICY_LOOKUP,
        policy_id="pol_meal_limit",
        as_of=world.as_of,
        include_superseded=False,
    )
    tool_turn = _run(sandbox, call)
    policy = tool_turn.tool_result.result["policy"]  # type: ignore[index]
    final = FinalAnswer(
        kind=FinalAnswerKind.ANSWER,
        text=(
            f"Use current policy {policy['version']} effective {policy['effective_from']}; "
            "do not apply superseded v1."
        ),
        structured={
            "version": policy["version"],
            "action": policy["action"],
            "reject_stale": True,
        },
        confidence=0.93,
        evidence=tool_turn.tool_result.provenance,  # type: ignore[union-attr]
    )
    return _finalize(
        user_goal="Apply the meal policy as of today. Ignore outdated versions.",
        turns_body=[
            TrajectoryTurn(turn_index=1, role=TurnRole.ASSISTANT, tool_call=call),
            tool_turn,
        ],
        final=final,
        required_tools=(ToolName.POLICY_LOOKUP,),
        gold_tool_calls=(call,),
        available_tools=available_tools,
        case_family=CaseFamily.STALE_POLICY,
        template_family="policy_current_only_v1",
    )


def _ambiguous_merchant(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    available_tools: tuple[ToolName, ...],
) -> OracleEpisode:
    call = _call(
        "c1",
        ToolName.LEDGER_QUERY,
        account_code=world.ledger[0].account_code,
        period=world.ledger[0].period,
        merchant_id=world.merchants[0].merchant_id,
    )
    tool_turn = _run(sandbox, call)
    names = [m.legal_name for m in world.merchants]
    final = FinalAnswer(
        kind=FinalAnswerKind.ANSWER,
        text=(
            "Merchant name is ambiguous between "
            + " and ".join(names)
            + f"; using merchant_id {world.merchants[0].merchant_id} for the posted row."
        ),
        structured={
            "resolved_merchant_id": world.merchants[0].merchant_id,
            "candidates": names,
            "ambiguous": True,
        },
        confidence=0.7,
        evidence=tool_turn.tool_result.provenance,  # type: ignore[union-attr]
    )
    return _finalize(
        user_goal="How much did we spend at Harbor Travel?",
        turns_body=[
            TrajectoryTurn(turn_index=1, role=TurnRole.ASSISTANT, tool_call=call),
            tool_turn,
        ],
        final=final,
        required_tools=(ToolName.LEDGER_QUERY,),
        gold_tool_calls=(call,),
        available_tools=available_tools,
        case_family=CaseFamily.AMBIGUOUS_MERCHANT,
        template_family="merchant_disambiguation_v1",
    )


def _multi_step_reconciliation(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    available_tools: tuple[ToolName, ...],
) -> OracleEpisode:
    book_id = world.book_entries[0].book_id
    bank_id = world.bank_events[0].bank_id
    match_call = _call(
        "c1",
        ToolName.TRANSACTION_MATCHING,
        book_ids=[book_id],
        bank_ids=[bank_id],
        tolerance_minor=0,
    )
    match_turn = _run(sandbox, match_call)
    calc_call = _call(
        "c2",
        ToolName.CALCULATOR,
        op="abs_diff",
        operands_minor=[
            world.book_entries[0].amount_minor,
            world.bank_events[0].amount_minor,
        ],
    )
    calc_turn = _run(sandbox, calc_call)
    matched = match_turn.tool_result.result["matched"]  # type: ignore[index]
    final = FinalAnswer(
        kind=FinalAnswerKind.ANSWER,
        text=f"Book {book_id} matches bank {bank_id}; matched={matched}.",
        structured={
            "matched": matched,
            "book_id": book_id,
            "bank_id": bank_id,
            "difference_minor": match_turn.tool_result.result["difference_minor"],  # type: ignore[index]
        },
        confidence=0.94,
        evidence=match_turn.tool_result.provenance + calc_turn.tool_result.provenance,  # type: ignore[operator]
    )
    return _finalize(
        user_goal="Reconcile the open book entry to the bank deposit.",
        turns_body=[
            TrajectoryTurn(turn_index=1, role=TurnRole.ASSISTANT, tool_call=match_call),
            match_turn,
            TrajectoryTurn(turn_index=3, role=TurnRole.ASSISTANT, tool_call=calc_call),
            calc_turn,
        ],
        final=final,
        required_tools=(ToolName.TRANSACTION_MATCHING, ToolName.CALCULATOR),
        gold_tool_calls=(match_call, calc_call),
        available_tools=available_tools,
        case_family=CaseFamily.MULTI_STEP_RECONCILIATION,
        template_family="match_then_diff_v1",
        max_tool_calls=4,
    )


def _arithmetic_trap(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    available_tools: tuple[ToolName, ...],
) -> OracleEpisode:
    drivers_call = _call(
        "c1",
        ToolName.VARIANCE_DRILL_DOWN,
        period=world.variance_drivers[0].period,
        account_code=world.variance_drivers[0].account_code,
        top_k=2,
    )
    drivers_turn = _run(sandbox, drivers_call)
    impacts = [d["impact_minor"] for d in drivers_turn.tool_result.result["drivers"]]  # type: ignore[index]
    calc_call = _call("c2", ToolName.CALCULATOR, op="add", operands_minor=impacts)
    calc_turn = _run(sandbox, calc_call)
    total = calc_turn.tool_result.result["result_minor"]  # type: ignore[index]
    # Trap: sign-preserving sum, not sum of absolutes.
    final = FinalAnswer(
        kind=FinalAnswerKind.ANSWER,
        text=f"Signed variance total is {total} minor units (do not sum absolute impacts).",
        structured={"total_impact_minor": total, "trap": "signed_not_abs"},
        confidence=0.91,
        evidence=calc_turn.tool_result.provenance,  # type: ignore[union-attr]
    )
    return _finalize(
        user_goal="Sum the top variance drivers for the account this period.",
        turns_body=[
            TrajectoryTurn(turn_index=1, role=TurnRole.ASSISTANT, tool_call=drivers_call),
            drivers_turn,
            TrajectoryTurn(turn_index=3, role=TurnRole.ASSISTANT, tool_call=calc_call),
            calc_turn,
        ],
        final=final,
        required_tools=(ToolName.VARIANCE_DRILL_DOWN, ToolName.CALCULATOR),
        gold_tool_calls=(drivers_call, calc_call),
        available_tools=available_tools,
        case_family=CaseFamily.ARITHMETIC_TRAP,
        template_family="signed_variance_sum_v1",
    )


def _conflicting_evidence(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    available_tools: tuple[ToolName, ...],
) -> OracleEpisode:
    # Ledger amount vs bank descriptor amount conflict simulation: bank equals book,
    # but we force a policy/ledger conflict narrative via memo vs policy threshold.
    ledger_call = _call(
        "c1",
        ToolName.LEDGER_QUERY,
        account_code=world.ledger[0].account_code,
        period=world.ledger[0].period,
    )
    ledger_turn = _run(sandbox, ledger_call)
    policy_call = _call(
        "c2",
        ToolName.POLICY_LOOKUP,
        policy_id="pol_meal_limit",
        as_of=world.as_of,
    )
    policy_turn = _run(sandbox, policy_call)
    amount = world.ledger[0].amount_minor
    action = policy_turn.tool_result.result["policy"]["action"]  # type: ignore[index]
    final = FinalAnswer(
        kind=FinalAnswerKind.ANSWER,
        text=(
            f"Ledger amount {amount} conflicts with a naive approve assumption; "
            f"policy requires {action}."
        ),
        structured={
            "amount_minor": amount,
            "policy_action": action,
            "conflict": "amount_vs_policy_threshold",
        },
        confidence=0.8,
        evidence=ledger_turn.tool_result.provenance + policy_turn.tool_result.provenance,  # type: ignore[operator]
    )
    return _finalize(
        user_goal="Should this meal expense be auto-approved given the ledger and policy?",
        turns_body=[
            TrajectoryTurn(turn_index=1, role=TurnRole.ASSISTANT, tool_call=ledger_call),
            ledger_turn,
            TrajectoryTurn(turn_index=3, role=TurnRole.ASSISTANT, tool_call=policy_call),
            policy_turn,
        ],
        final=final,
        required_tools=(ToolName.LEDGER_QUERY, ToolName.POLICY_LOOKUP),
        gold_tool_calls=(ledger_call, policy_call),
        available_tools=available_tools,
        case_family=CaseFamily.CONFLICTING_EVIDENCE,
        template_family="ledger_policy_conflict_v1",
    )


def _refusal_missing_data(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    available_tools: tuple[ToolName, ...],
) -> OracleEpisode:
    del sandbox  # refusal path must not invent tool-backed facts
    missing = world.missing_fields or ("merchant_id",)
    final = FinalAnswer(
        kind=FinalAnswerKind.REFUSAL,
        text=(
            "I cannot answer because required fields are missing: "
            + ", ".join(missing)
            + ". Provide the missing identifiers and retry."
        ),
        structured={"missing_fields": list(missing), "refused": True},
        confidence=1.0,
        evidence=(),
    )
    return _finalize(
        user_goal="Post and approve the expense for the unnamed merchant.",
        turns_body=[],
        final=final,
        required_tools=(),
        gold_tool_calls=(),
        available_tools=available_tools,
        case_family=CaseFamily.REFUSAL_MISSING_DATA,
        template_family="refuse_missing_v1",
        max_tool_calls=0,
    )

