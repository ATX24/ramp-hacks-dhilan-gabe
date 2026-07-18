"""Versioned Finance Agent episode and trajectory contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, StrictStr, field_validator, model_validator

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.hashing import NonNegativeSafeInt, SafeInt
from distillery.contracts.ids import ExampleId, GroupId, WorldId
from distillery.contracts.tasks import Difficulty, LabelSource, SplitName

SCHEMA_VERSION_FINANCE_AGENT: Literal["finance_agent.v1"] = "finance_agent.v1"
TECHNIQUE_ID_AGENT_TRAJECTORY: Literal["agent_trajectory.v1"] = "agent_trajectory.v1"

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
BoundedStr = Annotated[StrictStr, Field(min_length=1, max_length=256)]
BoundedText = Annotated[StrictStr, Field(min_length=1, max_length=4_096)]


class ToolName(StrEnum):
    CHART_OF_ACCOUNTS_LOOKUP = "chart_of_accounts_lookup"
    POLICY_LOOKUP = "policy_lookup"
    LEDGER_QUERY = "ledger_query"
    CALCULATOR = "calculator"
    TRANSACTION_MATCHING = "transaction_matching"
    VARIANCE_DRILL_DOWN = "variance_drill_down"


class CaseFamily(StrEnum):
    WRONG_TOOL = "wrong_tool"
    CORRECT_TOOL_WRONG_ARGS = "correct_tool_wrong_args"
    STALE_POLICY = "stale_policy"
    AMBIGUOUS_MERCHANT = "ambiguous_merchant"
    MULTI_STEP_RECONCILIATION = "multi_step_reconciliation"
    ARITHMETIC_TRAP = "arithmetic_trap"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    REFUSAL_MISSING_DATA = "refusal_missing_data"
    HAPPY_PATH = "happy_path"


class TurnRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinalAnswerKind(StrEnum):
    ANSWER = "answer"
    REFUSAL = "refusal"


class ProvenanceRef(FrozenModel):
    source_id: BoundedStr
    field: BoundedStr
    value: BoundedStr


class ToolCall(FrozenModel):
    """Assistant tool invocation with bounded arguments."""

    call_id: BoundedStr
    tool: ToolName
    arguments: FrozenJsonObject

    @field_validator("arguments")
    @classmethod
    def _bounded_argument_count(cls, value: Any) -> Any:
        if len(value) > 16:
            raise ValueError("tool arguments exceed bound of 16 keys")
        return value


class ToolResult(FrozenModel):
    call_id: BoundedStr
    tool: ToolName
    ok: bool
    result: FrozenJsonObject
    provenance: tuple[ProvenanceRef, ...] = ()
    error_code: StrictStr | None = None

    @model_validator(mode="after")
    def _error_consistency(self) -> ToolResult:
        if self.ok and self.error_code is not None:
            raise ValueError("ok tool result cannot carry error_code")
        if not self.ok and not self.error_code:
            raise ValueError("failed tool result requires error_code")
        return self


class FinalAnswer(FrozenModel):
    kind: FinalAnswerKind
    text: BoundedText
    structured: FrozenJsonObject = Field(default_factory=dict)
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    evidence: tuple[ProvenanceRef, ...] = ()


class TrajectoryTurn(FrozenModel):
    turn_index: NonNegativeSafeInt
    role: TurnRole
    text: StrictStr | None = Field(default=None, max_length=4_096)
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    final_answer: FinalAnswer | None = None

    @model_validator(mode="after")
    def _role_payload(self) -> TrajectoryTurn:
        if self.role is TurnRole.USER:
            if not self.text or self.tool_call or self.tool_result or self.final_answer:
                raise ValueError("user turn requires text only")
        elif self.role is TurnRole.ASSISTANT:
            has_call = self.tool_call is not None
            has_final = self.final_answer is not None
            if has_call == has_final:
                raise ValueError("assistant turn requires exactly one of tool_call or final_answer")
            if self.tool_result is not None:
                raise ValueError("assistant turn cannot carry tool_result")
            if has_call and self.text:
                raise ValueError("assistant tool_call turn must not include free text")
        elif self.role is TurnRole.TOOL:
            if self.tool_result is None or self.tool_call or self.final_answer or self.text:
                raise ValueError("tool turn requires tool_result only")
        return self


class AgentTrajectory(FrozenModel):
    schema_version: Literal["finance_agent.trajectory.v1"] = "finance_agent.trajectory.v1"
    turns: tuple[TrajectoryTurn, ...] = Field(min_length=2, max_length=64)

    @model_validator(mode="after")
    def _trajectory_shape(self) -> AgentTrajectory:
        if self.turns[0].role is not TurnRole.USER:
            raise ValueError("trajectory must start with a user turn")
        if self.turns[-1].role is not TurnRole.ASSISTANT or self.turns[-1].final_answer is None:
            raise ValueError("trajectory must end with an assistant final_answer")
        indices = [turn.turn_index for turn in self.turns]
        if indices != list(range(len(self.turns))):
            raise ValueError("turn_index values must be contiguous from 0")
        open_calls: dict[str, ToolName] = {}
        for turn in self.turns:
            if turn.tool_call is not None:
                if turn.tool_call.call_id in open_calls:
                    raise ValueError(f"duplicate call_id {turn.tool_call.call_id}")
                open_calls[turn.tool_call.call_id] = turn.tool_call.tool
            if turn.tool_result is not None:
                expected = open_calls.pop(turn.tool_result.call_id, None)
                if expected is None:
                    raise ValueError(f"tool result for unknown call_id {turn.tool_result.call_id}")
                if expected is not turn.tool_result.tool:
                    raise ValueError("tool result tool does not match open call")
        if open_calls:
            raise ValueError(f"unresolved tool calls: {sorted(open_calls)}")
        return self


class OracleMeta(FrozenModel):
    generator_revision: Literal["finance_agent.v1"] = "finance_agent.v1"
    latent_state_hash: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    technique_id: Literal["agent_trajectory.v1"] = TECHNIQUE_ID_AGENT_TRAJECTORY


class AgentProvenance(FrozenModel):
    split: SplitName
    case_family: CaseFamily
    template_family: BoundedStr
    label_source: LabelSource
    held_out_tools: tuple[ToolName, ...] = ()
    held_out_domains: tuple[BoundedStr, ...] = ()


class ExpectedAgentOutput(FrozenModel):
    schema_version: Literal["finance_agent.output.v1"] = "finance_agent.output.v1"
    final_answer: FinalAnswer
    required_tools: tuple[ToolName, ...]
    forbidden_tools: tuple[ToolName, ...] = ()
    max_tool_calls: NonNegativeSafeInt = 8
    gold_tool_calls: tuple[ToolCall, ...] = ()


class AgentEpisodeEnvelope(FrozenModel):
    """Canonical episode envelope for finance_agent.v1 (not finance_world)."""

    schema_version: Literal["finance_agent.v1"] = SCHEMA_VERSION_FINANCE_AGENT
    example_id: ExampleId
    world_id: WorldId
    group_id: GroupId
    difficulty: Difficulty
    case_family: CaseFamily
    user_goal: BoundedText
    available_tools: tuple[ToolName, ...] = Field(min_length=1)
    trajectory: AgentTrajectory
    expected_output: ExpectedAgentOutput
    oracle: OracleMeta
    provenance: AgentProvenance
    estimated_latency_ms: NonNegativeSafeInt = 0
    estimated_cost_usd_micros: NonNegativeSafeInt = 0

    @model_validator(mode="after")
    def _consistency(self) -> AgentEpisodeEnvelope:
        if self.case_family != self.provenance.case_family:
            raise ValueError("case_family must match provenance.case_family")
        final = self.trajectory.turns[-1].final_answer
        assert final is not None
        if final.model_dump(mode="json") != self.expected_output.final_answer.model_dump(
            mode="json"
        ):
            raise ValueError("trajectory final_answer must match expected_output.final_answer")
        available = set(self.available_tools)
        for call in self.expected_output.gold_tool_calls:
            if call.tool not in available:
                raise ValueError(f"gold tool {call.tool} not in available_tools")
        for tool in self.expected_output.required_tools:
            if tool not in available:
                raise ValueError(f"required tool {tool} not in available_tools")
        return self


class MoneyMinor(FrozenModel):
    amount_minor: SafeInt
    currency: Literal["USD"] = "USD"


__all__ = [
    "SCHEMA_VERSION_FINANCE_AGENT",
    "TECHNIQUE_ID_AGENT_TRAJECTORY",
    "AgentEpisodeEnvelope",
    "AgentProvenance",
    "AgentTrajectory",
    "CaseFamily",
    "ExpectedAgentOutput",
    "FinalAnswer",
    "FinalAnswerKind",
    "MoneyMinor",
    "OracleMeta",
    "ProvenanceRef",
    "ToolCall",
    "ToolName",
    "ToolResult",
    "TrajectoryTurn",
    "TurnRole",
]
