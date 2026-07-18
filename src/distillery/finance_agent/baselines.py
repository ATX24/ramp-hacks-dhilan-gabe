"""Non-oracle baselines used to ensure metrics do not reward trivial behavior."""

from __future__ import annotations

from distillery.finance_agent.contracts import (
    AgentEpisodeEnvelope,
    AgentTrajectory,
    FinalAnswer,
    FinalAnswerKind,
    ToolCall,
    ToolResult,
    TrajectoryTurn,
    TurnRole,
)


def always_refuse_baseline(example: AgentEpisodeEnvelope) -> AgentTrajectory:
    initial = example.model_input.user_messages[0].text
    return AgentTrajectory(
        turns=(
            TrajectoryTurn(turn_index=0, role=TurnRole.USER, text=initial),
            TrajectoryTurn(
                turn_index=1,
                role=TurnRole.ASSISTANT,
                final_answer=FinalAnswer(
                    kind=FinalAnswerKind.REFUSAL,
                    text="I cannot answer this request.",
                    structured={"refused": True},
                    confidence=0.0,
                ),
            ),
        )
    )


def first_tool_invalid_args_baseline(example: AgentEpisodeEnvelope) -> AgentTrajectory:
    initial = example.model_input.user_messages[0].text
    tool = example.model_input.tools[0].name
    call = ToolCall.seal(call_id="baseline_c1", tool=tool, arguments={})
    result = ToolResult.seal(
        call_id=call.call_id,
        tool=tool,
        ok=False,
        result={"message": "baseline supplied no arguments"},
        error_code="INVALID_ARGUMENTS",
    )
    return AgentTrajectory(
        turns=(
            TrajectoryTurn(turn_index=0, role=TurnRole.USER, text=initial),
            TrajectoryTurn(turn_index=1, role=TurnRole.ASSISTANT, tool_call=call),
            TrajectoryTurn(turn_index=2, role=TurnRole.TOOL, tool_result=result),
            TrajectoryTurn(
                turn_index=3,
                role=TurnRole.ASSISTANT,
                final_answer=FinalAnswer(
                    kind=FinalAnswerKind.REFUSAL,
                    text="The first tool call failed, so I cannot answer.",
                    structured={"refused": True},
                    confidence=0.0,
                ),
            ),
        )
    )


__all__ = ["always_refuse_baseline", "first_tool_invalid_args_baseline"]
