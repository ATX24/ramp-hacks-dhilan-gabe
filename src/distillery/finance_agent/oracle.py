"""Deterministic oracle trajectories, including explicit failure-and-recovery cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from distillery.finance_agent.contracts import (
    AgentTrajectory,
    CaseFamily,
    ExpectedAgentOutput,
    FinalAnswer,
    FinalAnswerKind,
    ResultBinding,
    ToolCall,
    ToolName,
    TrajectoryTurn,
    TurnRole,
)
from distillery.finance_agent.prompts import PromptPlan
from distillery.finance_agent.sandbox import FinanceAgentSandbox
from distillery.finance_agent.world import AgentWorld


@dataclass(frozen=True)
class OracleEpisode:
    trajectory: AgentTrajectory
    expected_output: ExpectedAgentOutput
    case_family: CaseFamily


def solve_case(
    world: AgentWorld,
    *,
    case_family: CaseFamily,
    available_tools: tuple[ToolName, ...],
    prompt_plan: PromptPlan,
) -> OracleEpisode:
    sandbox = FinanceAgentSandbox(world, allowed_tools=available_tools)
    builders = {
        CaseFamily.HAPPY_PATH: _happy_path,
        CaseFamily.WRONG_TOOL: _wrong_tool_recovery,
        CaseFamily.CORRECT_TOOL_WRONG_ARGS: _wrong_arguments_recovery,
        CaseFamily.STALE_POLICY: _stale_policy_recovery,
        CaseFamily.AMBIGUOUS_MERCHANT: _ambiguous_merchant,
        CaseFamily.MULTI_STEP_RECONCILIATION: _multi_step_reconciliation,
        CaseFamily.ARITHMETIC_TRAP: _arithmetic_trap,
        CaseFamily.CONFLICTING_EVIDENCE: _conflicting_evidence,
        CaseFamily.REFUSAL_MISSING_DATA: _refusal_missing_data,
    }
    return builders[case_family](world, sandbox, prompt_plan)


def _call(call_id: str, tool: ToolName, **arguments: Any) -> ToolCall:
    return ToolCall.seal(call_id=call_id, tool=tool, arguments=arguments)


def _user(text: str) -> TrajectoryTurn:
    return TrajectoryTurn(turn_index=0, role=TurnRole.USER, text=text)


def _assistant_text(text: str) -> TrajectoryTurn:
    return TrajectoryTurn(turn_index=0, role=TurnRole.ASSISTANT, text=text)


def _call_pair(
    sandbox: FinanceAgentSandbox,
    call: ToolCall,
) -> tuple[TrajectoryTurn, TrajectoryTurn]:
    result = sandbox.execute(
        call_id=call.call_id,
        tool=call.tool,
        arguments=call.arguments,
    )
    return (
        TrajectoryTurn(turn_index=0, role=TurnRole.ASSISTANT, tool_call=call),
        TrajectoryTurn(turn_index=0, role=TurnRole.TOOL, tool_result=result),
    )


def _at(value: Any, *path: str) -> Any:
    current = value
    for component in path:
        if isinstance(current, (list, tuple)):
            current = current[int(component)]
        else:
            current = current[component]
    return current


def _finalize(
    *,
    prompt_plan: PromptPlan,
    turns: list[TrajectoryTurn],
    final: FinalAnswer,
    bindings: tuple[ResultBinding, ...] = (),
    max_tool_calls: int = 8,
    case_family: CaseFamily,
) -> OracleEpisode:
    turns.append(TrajectoryTurn(turn_index=0, role=TurnRole.ASSISTANT, final_answer=final))
    reindexed = tuple(
        turn.model_copy(update={"turn_index": index}) for index, turn in enumerate(turns)
    )
    trajectory = AgentTrajectory(turns=reindexed)
    expected = ExpectedAgentOutput(
        final_answer=final,
        result_bindings=bindings,
        max_tool_calls=max_tool_calls,
    )
    expected_users = tuple(
        (message.turn_index, message.text) for message in prompt_plan.user_messages
    )
    actual_users = tuple(
        (turn.turn_index, turn.text) for turn in trajectory.turns if turn.role is TurnRole.USER
    )
    if expected_users != actual_users:
        raise ValueError("oracle trajectory user turns drifted from model-visible prompts")
    return OracleEpisode(
        trajectory=trajectory,
        expected_output=expected,
        case_family=case_family,
    )


def _happy_path(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    prompt_plan: PromptPlan,
) -> OracleEpisode:
    call = _call(
        "c1",
        ToolName.LEDGER_QUERY,
        account_code=world.accounts[0].code,
        period=world.period,
    )
    call_turn, result_turn = _call_pair(sandbox, call)
    assert result_turn.tool_result is not None
    total = result_turn.tool_result.result["total_amount_minor"]
    final = FinalAnswer(
        kind=FinalAnswerKind.ANSWER,
        text=(
            f"Account {world.accounts[0].code} has {total} minor units posted in {world.period}."
        ),
        structured={
            "account_code": world.accounts[0].code,
            "period": world.period,
            "amount_minor": total,
        },
        confidence=0.95,
        evidence=result_turn.tool_result.provenance,
    )
    return _finalize(
        prompt_plan=prompt_plan,
        turns=[_user(prompt_plan.user_messages[0].text), call_turn, result_turn],
        final=final,
        bindings=(
            ResultBinding(
                answer_path=("amount_minor",),
                call_id="c1",
                result_path=("total_amount_minor",),
            ),
        ),
        case_family=CaseFamily.HAPPY_PATH,
    )


def _wrong_tool_recovery(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    prompt_plan: PromptPlan,
) -> OracleEpisode:
    wrong = _call("c1", ToolName.CALCULATOR, op="add", operands=[1, 2])
    wrong_turn, wrong_result = _call_pair(sandbox, wrong)
    correct = _call(
        "c2",
        ToolName.POLICY_LOOKUP,
        policy_id=world.policies[-1].policy_id,
        as_of=world.as_of,
        include_history=False,
    )
    correct_turn, correct_result = _call_pair(sandbox, correct)
    assert correct_result.tool_result is not None
    policy = correct_result.tool_result.result["policy"]
    final = FinalAnswer(
        kind=FinalAnswerKind.ANSWER,
        text=(
            f"The calculator result was irrelevant. Policy {policy['policy_id']} "
            f"{policy['version']} is effective on {world.as_of}."
        ),
        structured={
            "recovered_from_wrong_tool": True,
            "policy_id": policy["policy_id"],
            "version": policy["version"],
        },
        confidence=0.9,
        evidence=correct_result.tool_result.provenance,
    )
    return _finalize(
        prompt_plan=prompt_plan,
        turns=[
            _user(prompt_plan.user_messages[0].text),
            wrong_turn,
            wrong_result,
            correct_turn,
            correct_result,
        ],
        final=final,
        bindings=(
            ResultBinding(
                answer_path=("version",),
                call_id="c2",
                result_path=("policy", "version"),
            ),
        ),
        case_family=CaseFamily.WRONG_TOOL,
    )


def _wrong_arguments_recovery(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    prompt_plan: PromptPlan,
) -> OracleEpisode:
    wrong = _call(
        "c1",
        ToolName.CHART_OF_ACCOUNTS_LOOKUP,
        query=world.accounts[0].name,
        account_code="9999",
    )
    wrong_turn, wrong_result = _call_pair(sandbox, wrong)
    assert wrong_result.tool_result is not None and not wrong_result.tool_result.ok
    correct = _call(
        "c2",
        ToolName.CHART_OF_ACCOUNTS_LOOKUP,
        query=world.accounts[0].name,
        account_code=world.accounts[0].code,
    )
    correct_turn, correct_result = _call_pair(sandbox, correct)
    assert correct_result.tool_result is not None
    match = correct_result.tool_result.result["matches"][0]
    final = FinalAnswer(
        kind=FinalAnswerKind.ANSWER,
        text=(
            f"Account 9999 did not match. The corrected account is {match['code']} "
            f"({match['name']})."
        ),
        structured={
            "recovered_from_wrong_arguments": True,
            "account_code": match["code"],
            "account_name": match["name"],
        },
        confidence=0.94,
        evidence=correct_result.tool_result.provenance,
    )
    return _finalize(
        prompt_plan=prompt_plan,
        turns=[
            _user(prompt_plan.user_messages[0].text),
            wrong_turn,
            wrong_result,
            correct_turn,
            correct_result,
        ],
        final=final,
        bindings=(
            ResultBinding(
                answer_path=("account_code",),
                call_id="c2",
                result_path=("matches", "0", "code"),
            ),
        ),
        case_family=CaseFamily.CORRECT_TOOL_WRONG_ARGS,
    )


def _stale_policy_recovery(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    prompt_plan: PromptPlan,
) -> OracleEpisode:
    stale = _call(
        "c1",
        ToolName.POLICY_LOOKUP,
        policy_id=world.policies[-1].policy_id,
        as_of=world.historical_policy_date,
        include_history=True,
    )
    stale_turn, stale_result = _call_pair(sandbox, stale)
    assert stale_result.tool_result is not None
    current = _call(
        "c2",
        ToolName.POLICY_LOOKUP,
        policy_id=world.policies[-1].policy_id,
        as_of=world.as_of,
        include_history=False,
    )
    current_turn, current_result = _call_pair(sandbox, current)
    assert current_result.tool_result is not None
    old_policy = stale_result.tool_result.result["policy"]
    policy = current_result.tool_result.result["policy"]
    final = FinalAnswer(
        kind=FinalAnswerKind.ANSWER,
        text=(
            f"{old_policy['version']} was valid on {world.historical_policy_date}; "
            f"for {world.as_of}, use {policy['version']}."
        ),
        structured={
            "historical_version": old_policy["version"],
            "current_version": policy["version"],
            "as_of": world.as_of,
        },
        confidence=0.96,
        evidence=current_result.tool_result.provenance,
    )
    return _finalize(
        prompt_plan=prompt_plan,
        turns=[
            _user(prompt_plan.user_messages[0].text),
            stale_turn,
            stale_result,
            current_turn,
            current_result,
        ],
        final=final,
        bindings=(
            ResultBinding(
                answer_path=("historical_version",),
                call_id="c1",
                result_path=("policy", "version"),
            ),
            ResultBinding(
                answer_path=("current_version",),
                call_id="c2",
                result_path=("policy", "version"),
            ),
        ),
        case_family=CaseFamily.STALE_POLICY,
    )


def _ambiguous_merchant(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    prompt_plan: PromptPlan,
) -> OracleEpisode:
    if len(prompt_plan.user_messages) != 2:
        raise ValueError("ambiguous merchant case requires a user clarification turn")
    clarification = _assistant_text(
        "That descriptor maps to multiple public merchant candidates. Which merchant_id "
        "should I use?"
    )
    chosen = world.merchants[0]
    call = _call(
        "c1",
        ToolName.LEDGER_QUERY,
        account_code=world.accounts[0].code,
        period=world.period,
        merchant_id=chosen.merchant_id,
    )
    call_turn, result_turn = _call_pair(sandbox, call)
    assert result_turn.tool_result is not None
    total = result_turn.tool_result.result["total_amount_minor"]
    final = FinalAnswer(
        kind=FinalAnswerKind.ANSWER,
        text=f"Spend for {chosen.merchant_id} is {total} minor units.",
        structured={
            "merchant_id": chosen.merchant_id,
            "amount_minor": total,
            "clarification_used": True,
        },
        confidence=0.93,
        evidence=result_turn.tool_result.provenance,
    )
    return _finalize(
        prompt_plan=prompt_plan,
        turns=[
            _user(prompt_plan.user_messages[0].text),
            clarification,
            _user(prompt_plan.user_messages[1].text),
            call_turn,
            result_turn,
        ],
        final=final,
        bindings=(
            ResultBinding(
                answer_path=("amount_minor",),
                call_id="c1",
                result_path=("total_amount_minor",),
            ),
        ),
        case_family=CaseFamily.AMBIGUOUS_MERCHANT,
    )


def _multi_step_reconciliation(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    prompt_plan: PromptPlan,
) -> OracleEpisode:
    match = _call(
        "c1",
        ToolName.TRANSACTION_MATCHING,
        book_ids=[entry.book_id for entry in world.book_entries],
        bank_ids=[event.bank_id for event in world.bank_events],
        tolerance_minor=0,
    )
    match_turn, match_result = _call_pair(sandbox, match)
    assert match_result.tool_result is not None
    match_payload = match_result.tool_result.result
    calculator = _call(
        "c2",
        ToolName.CALCULATOR,
        op="abs_diff",
        operands=[
            match_payload["book_sum_minor"],
            match_payload["bank_sum_minor"],
        ],
    )
    calculator_turn, calculator_result = _call_pair(sandbox, calculator)
    assert calculator_result.tool_result is not None
    verified = calculator_result.tool_result.result["result"]
    final = FinalAnswer(
        kind=FinalAnswerKind.ANSWER,
        text=(
            f"The transactions match with signed difference "
            f"{match_payload['difference_minor']} and absolute difference {verified}."
        ),
        structured={
            "matched": match_payload["matched"],
            "difference_minor": match_payload["difference_minor"],
            "verified_abs_difference_minor": verified,
        },
        confidence=0.97,
        evidence=(match_result.tool_result.provenance + calculator_result.tool_result.provenance),
    )
    return _finalize(
        prompt_plan=prompt_plan,
        turns=[
            _user(prompt_plan.user_messages[0].text),
            match_turn,
            match_result,
            calculator_turn,
            calculator_result,
        ],
        final=final,
        bindings=(
            ResultBinding(
                answer_path=("matched",),
                call_id="c1",
                result_path=("matched",),
            ),
            ResultBinding(
                answer_path=("difference_minor",),
                call_id="c1",
                result_path=("difference_minor",),
            ),
            ResultBinding(
                answer_path=("verified_abs_difference_minor",),
                call_id="c2",
                result_path=("result",),
            ),
        ),
        max_tool_calls=2,
        case_family=CaseFamily.MULTI_STEP_RECONCILIATION,
    )


def _arithmetic_trap(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    prompt_plan: PromptPlan,
) -> OracleEpisode:
    drill = _call(
        "c1",
        ToolName.VARIANCE_DRILL_DOWN,
        period=world.period,
        account_code=world.accounts[0].code,
        top_k=len(world.variance_drivers),
    )
    drill_turn, drill_result = _call_pair(sandbox, drill)
    assert drill_result.tool_result is not None
    impacts = [driver["impact_minor"] for driver in drill_result.tool_result.result["drivers"]]
    calculator = _call("c2", ToolName.CALCULATOR, op="add", operands=impacts)
    calculator_turn, calculator_result = _call_pair(sandbox, calculator)
    assert calculator_result.tool_result is not None
    signed_total = calculator_result.tool_result.result["result"]
    full_total = drill_result.tool_result.result["full_period_impact_minor"]
    if signed_total != full_total:
        raise ValueError("oracle variance arithmetic does not close")
    final = FinalAnswer(
        kind=FinalAnswerKind.ANSWER,
        text=f"Signed full-period variance is {signed_total} minor units.",
        structured={
            "signed_total_minor": signed_total,
            "full_period_impact_minor": full_total,
            "used_absolute_values": False,
        },
        confidence=0.97,
        evidence=(drill_result.tool_result.provenance + calculator_result.tool_result.provenance),
    )
    return _finalize(
        prompt_plan=prompt_plan,
        turns=[
            _user(prompt_plan.user_messages[0].text),
            drill_turn,
            drill_result,
            calculator_turn,
            calculator_result,
        ],
        final=final,
        bindings=(
            ResultBinding(
                answer_path=("signed_total_minor",),
                call_id="c2",
                result_path=("result",),
            ),
            ResultBinding(
                answer_path=("full_period_impact_minor",),
                call_id="c1",
                result_path=("full_period_impact_minor",),
            ),
        ),
        max_tool_calls=2,
        case_family=CaseFamily.ARITHMETIC_TRAP,
    )


def _conflicting_evidence(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    prompt_plan: PromptPlan,
) -> OracleEpisode:
    ledger = _call(
        "c1",
        ToolName.LEDGER_QUERY,
        account_code=world.accounts[0].code,
        period=world.period,
    )
    ledger_turn, ledger_result = _call_pair(sandbox, ledger)
    policy = _call(
        "c2",
        ToolName.POLICY_LOOKUP,
        policy_id=world.policies[-1].policy_id,
        as_of=world.as_of,
        include_history=False,
    )
    policy_turn, policy_result = _call_pair(sandbox, policy)
    assert ledger_result.tool_result is not None
    assert policy_result.tool_result is not None
    amount = ledger_result.tool_result.result["total_amount_minor"]
    policy_payload = policy_result.tool_result.result["policy"]
    threshold = policy_payload["threshold_minor"]
    action = (
        policy_payload["action_above"]
        if amount > threshold
        else policy_payload["action_at_or_below"]
    )
    conflict = amount > threshold and action != "approve"
    if not conflict:
        raise ValueError("conflicting-evidence world must actually cross the threshold")
    final = FinalAnswer(
        kind=FinalAnswerKind.ANSWER,
        text=(
            f"The posted total {amount} exceeds threshold {threshold}; policy action "
            f"is {action}, so auto-approval conflicts with the evidence."
        ),
        structured={
            "amount_minor": amount,
            "threshold_minor": threshold,
            "policy_action": action,
            "conflict": True,
        },
        confidence=0.96,
        evidence=ledger_result.tool_result.provenance + policy_result.tool_result.provenance,
    )
    return _finalize(
        prompt_plan=prompt_plan,
        turns=[
            _user(prompt_plan.user_messages[0].text),
            ledger_turn,
            ledger_result,
            policy_turn,
            policy_result,
        ],
        final=final,
        bindings=(
            ResultBinding(
                answer_path=("amount_minor",),
                call_id="c1",
                result_path=("total_amount_minor",),
            ),
            ResultBinding(
                answer_path=("threshold_minor",),
                call_id="c2",
                result_path=("policy", "threshold_minor"),
            ),
        ),
        max_tool_calls=2,
        case_family=CaseFamily.CONFLICTING_EVIDENCE,
    )


def _refusal_missing_data(
    world: AgentWorld,
    sandbox: FinanceAgentSandbox,
    prompt_plan: PromptPlan,
) -> OracleEpisode:
    del sandbox
    if not world.missing_fields:
        raise ValueError("refusal case requires explicit missing fields")
    final = FinalAnswer(
        kind=FinalAnswerKind.REFUSAL,
        text=(
            "I cannot post or approve this expense because required fields are "
            f"missing: {', '.join(world.missing_fields)}."
        ),
        structured={
            "refused": True,
            "missing_fields": list(world.missing_fields),
        },
        confidence=1.0,
    )
    return _finalize(
        prompt_plan=prompt_plan,
        turns=[_user(prompt_plan.user_messages[0].text)],
        final=final,
        max_tool_calls=0,
        case_family=CaseFamily.REFUSAL_MISSING_DATA,
    )


__all__ = ["OracleEpisode", "solve_case"]
