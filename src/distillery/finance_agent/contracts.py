"""Versioned, sealed contracts for Finance Agent inputs and gold trajectories."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    Field,
    JsonValue,
    StrictBool,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.hashing import (
    NonNegativeSafeInt,
    SafeInt,
    Sha256Hex,
    content_sha256,
)
from distillery.contracts.ids import ExampleId, GroupId, WorldId
from distillery.contracts.tasks import Difficulty, LabelSource, SplitName

SCHEMA_VERSION_FINANCE_AGENT: Literal["finance_agent.v1"] = "finance_agent.v1"
TECHNIQUE_ID_AGENT_TRAJECTORY: Literal["agent_trajectory.v1"] = "agent_trajectory.v1"

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
BoundedStr = Annotated[StrictStr, Field(min_length=1, max_length=256)]
BoundedText = Annotated[StrictStr, Field(min_length=1, max_length=8_192)]


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
    SYSTEM = "system"
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


class ToolDefinition(FrozenModel):
    name: ToolName
    description: BoundedText
    input_schema: FrozenJsonObject
    input_schema_sha256: Sha256Hex

    @model_validator(mode="after")
    def _schema_hash_matches(self) -> Self:
        expected = content_sha256(self.input_schema)
        if self.input_schema_sha256 != expected:
            raise ValueError("input_schema_sha256 does not match canonical input_schema")
        return self

    @classmethod
    def seal(
        cls,
        *,
        name: ToolName,
        description: str,
        input_schema: dict[str, JsonValue],
    ) -> ToolDefinition:
        return cls(
            name=name,
            description=description,
            input_schema=input_schema,
            input_schema_sha256=content_sha256(input_schema),
        )


class ToolCall(FrozenModel):
    """Assistant tool invocation with content-bound arguments."""

    call_id: BoundedStr
    tool: ToolName
    arguments: FrozenJsonObject
    arguments_sha256: Sha256Hex

    @field_validator("arguments")
    @classmethod
    def _bounded_argument_count(cls, value: Any) -> Any:
        if len(value) > 16:
            raise ValueError("tool arguments exceed bound of 16 keys")
        return value

    @model_validator(mode="after")
    def _hash_matches(self, info: ValidationInfo) -> Self:
        if not info.context or not info.context.get("skip_hash_validation", False):
            if self.arguments_sha256 != content_sha256(self.arguments):
                raise ValueError("arguments_sha256 does not match canonical arguments")
        return self

    @classmethod
    def seal(cls, *, call_id: str, tool: ToolName, arguments: dict[str, Any]) -> ToolCall:
        return cls(
            call_id=call_id,
            tool=tool,
            arguments=arguments,
            arguments_sha256=content_sha256(arguments),
        )


class ToolResult(FrozenModel):
    """A sealed sandbox result. Failures are values, not evaluator exceptions."""

    call_id: BoundedStr
    tool: ToolName
    ok: StrictBool
    result: FrozenJsonObject
    provenance: tuple[ProvenanceRef, ...] = ()
    error_code: StrictStr | None = None
    result_sha256: Sha256Hex

    @model_validator(mode="after")
    def _invariants(self, info: ValidationInfo) -> Self:
        if self.ok and self.error_code is not None:
            raise ValueError("ok tool result cannot carry error_code")
        if not self.ok and not self.error_code:
            raise ValueError("failed tool result requires error_code")
        if not info.context or not info.context.get("skip_hash_validation", False):
            if self.result_sha256 != content_sha256(self.canonical_payload()):
                raise ValueError("result_sha256 does not match canonical tool result")
        return self

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"result_sha256"})

    @classmethod
    def seal(
        cls,
        *,
        call_id: str,
        tool: ToolName,
        ok: bool,
        result: dict[str, Any],
        provenance: tuple[ProvenanceRef, ...] = (),
        error_code: str | None = None,
    ) -> ToolResult:
        provisional = cls.model_validate(
            {
                "call_id": call_id,
                "tool": tool,
                "ok": ok,
                "result": result,
                "provenance": provenance,
                "error_code": error_code,
                "result_sha256": "0" * 64,
            },
            context={"skip_hash_validation": True},
        )
        payload = provisional.canonical_payload()
        return cls.model_validate({**payload, "result_sha256": content_sha256(payload)})


class FinalAnswer(FrozenModel):
    kind: FinalAnswerKind
    text: BoundedText
    structured: FrozenJsonObject = Field(default_factory=dict)
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    evidence: tuple[ProvenanceRef, ...] = ()


class TrajectoryTurn(FrozenModel):
    turn_index: NonNegativeSafeInt
    role: TurnRole
    text: StrictStr | None = Field(default=None, max_length=8_192)
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    final_answer: FinalAnswer | None = None

    @model_validator(mode="after")
    def _role_payload(self) -> Self:
        if self.role is TurnRole.SYSTEM:
            raise ValueError("system prompt belongs in model_input, not the gold trajectory")
        if self.role is TurnRole.USER:
            if not self.text or self.tool_call or self.tool_result or self.final_answer:
                raise ValueError("user turn requires text only")
            return self
        if self.role is TurnRole.TOOL:
            if self.tool_result is None or self.tool_call or self.final_answer or self.text:
                raise ValueError("tool turn requires tool_result only")
            return self
        populated = sum(
            value is not None for value in (self.text, self.tool_call, self.final_answer)
        )
        if populated != 1 or self.tool_result is not None:
            raise ValueError(
                "assistant turn requires exactly one of text, tool_call, or final_answer"
            )
        return self


class AgentTrajectory(FrozenModel):
    schema_version: Literal["finance_agent.trajectory.v1"] = "finance_agent.trajectory.v1"
    turns: tuple[TrajectoryTurn, ...] = Field(min_length=2, max_length=96)

    @model_validator(mode="after")
    def _trajectory_shape(self) -> Self:
        if self.turns[0].role is not TurnRole.USER:
            raise ValueError("trajectory must start with a user turn")
        final_turn = self.turns[-1]
        if final_turn.role is not TurnRole.ASSISTANT or final_turn.final_answer is None:
            raise ValueError("trajectory must end with an assistant final_answer")
        indices = [turn.turn_index for turn in self.turns]
        if indices != list(range(len(self.turns))):
            raise ValueError("turn_index values must be contiguous from 0")
        for index, turn in enumerate(self.turns):
            if turn.final_answer is not None and index != len(self.turns) - 1:
                raise ValueError("final_answer may appear only on the final turn")
            if turn.tool_call is not None:
                if index + 1 >= len(self.turns):
                    raise ValueError("tool call requires an immediately following result")
                result = self.turns[index + 1].tool_result
                if result is None:
                    raise ValueError("tool call requires an immediately following result")
                if (
                    result.call_id != turn.tool_call.call_id
                    or result.tool is not turn.tool_call.tool
                ):
                    raise ValueError("tool result must match the immediately preceding call")
            if turn.tool_result is not None:
                if index == 0 or self.turns[index - 1].tool_call is None:
                    raise ValueError("tool result must immediately follow its tool call")
        return self

    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(turn.tool_call for turn in self.turns if turn.tool_call is not None)

    def tool_results(self) -> tuple[ToolResult, ...]:
        return tuple(turn.tool_result for turn in self.turns if turn.tool_result is not None)


class UserMessage(FrozenModel):
    turn_index: NonNegativeSafeInt
    text: BoundedText


class AgentModelInput(FrozenModel):
    """The only episode payload model/trainer/eval inference may read."""

    schema_version: Literal["finance_agent.model_input.v1"] = "finance_agent.model_input.v1"
    example_id: ExampleId
    system_prompt: BoundedText
    public_world: FrozenJsonObject
    user_messages: tuple[UserMessage, ...] = Field(min_length=1, max_length=16)
    tools: tuple[ToolDefinition, ...] = Field(min_length=1)
    system_prompt_sha256: Sha256Hex
    public_world_sha256: Sha256Hex
    tool_schemas_sha256: Sha256Hex
    model_input_sha256: Sha256Hex

    @model_validator(mode="after")
    def _hashes_match(self, info: ValidationInfo) -> Self:
        if info.context and info.context.get("skip_hash_validation", False):
            return self
        if self.system_prompt_sha256 != content_sha256(self.system_prompt):
            raise ValueError("system_prompt_sha256 mismatch")
        if self.public_world_sha256 != content_sha256(self.public_world):
            raise ValueError("public_world_sha256 mismatch")
        tools_payload = [tool.model_dump(mode="json") for tool in self.tools]
        if self.tool_schemas_sha256 != content_sha256(tools_payload):
            raise ValueError("tool_schemas_sha256 mismatch")
        if self.model_input_sha256 != content_sha256(self.canonical_payload()):
            raise ValueError("model_input_sha256 mismatch")
        return self

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"model_input_sha256"})

    @classmethod
    def seal(
        cls,
        *,
        example_id: str,
        system_prompt: str,
        public_world: dict[str, Any],
        user_messages: tuple[UserMessage, ...],
        tools: tuple[ToolDefinition, ...],
    ) -> AgentModelInput:
        tools_payload = [tool.model_dump(mode="json") for tool in tools]
        provisional = cls.model_validate(
            {
                "example_id": example_id,
                "system_prompt": system_prompt,
                "public_world": public_world,
                "user_messages": user_messages,
                "tools": tools,
                "system_prompt_sha256": content_sha256(system_prompt),
                "public_world_sha256": content_sha256(public_world),
                "tool_schemas_sha256": content_sha256(tools_payload),
                "model_input_sha256": "0" * 64,
            },
            context={"skip_hash_validation": True},
        )
        payload = provisional.canonical_payload()
        return cls.model_validate({**payload, "model_input_sha256": content_sha256(payload)})


class TeacherRolloutEvidence(FrozenModel):
    """Required evidence for future teacher-labeled trajectories."""

    model_id: BoundedStr
    model_revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    rollout_artifact_sha256: Sha256Hex
    license_id: BoundedStr
    license_text_sha256: Sha256Hex
    output_use_reviewed: Literal[True]
    attribution_disposition: Literal["not_required", "required_and_recorded"]
    attribution_text: StrictStr | None = Field(default=None, max_length=1_024)

    @model_validator(mode="after")
    def _attribution_recorded(self) -> Self:
        if self.attribution_disposition == "required_and_recorded" and not self.attribution_text:
            raise ValueError("required attribution must include attribution_text")
        if self.attribution_disposition == "not_required" and self.attribution_text:
            raise ValueError("not_required attribution must not include attribution_text")
        return self


class OracleMeta(FrozenModel):
    generator_revision: Literal["finance_agent.v2"] = "finance_agent.v2"
    latent_state_hash: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    label_source: LabelSource
    teacher_evidence: TeacherRolloutEvidence | None = None

    @model_validator(mode="after")
    def _label_evidence(self) -> Self:
        if self.label_source is LabelSource.ORACLE and self.teacher_evidence is not None:
            raise ValueError("oracle labels cannot carry teacher evidence")
        if self.label_source is LabelSource.TEACHER and self.teacher_evidence is None:
            raise ValueError("teacher labels require exact rollout/license evidence")
        if self.label_source not in {LabelSource.ORACLE, LabelSource.TEACHER}:
            raise ValueError("Finance Agent labels must be oracle or teacher")
        return self


class ResultBinding(FrozenModel):
    """Expected final-answer field copied/derived from a specific tool result."""

    answer_path: tuple[BoundedStr, ...] = Field(min_length=1)
    call_id: BoundedStr
    result_path: tuple[BoundedStr, ...] = Field(min_length=1)


class ExpectedAgentOutput(FrozenModel):
    schema_version: Literal["finance_agent.output.v2"] = "finance_agent.output.v2"
    final_answer: FinalAnswer
    result_bindings: tuple[ResultBinding, ...] = ()
    max_tool_calls: NonNegativeSafeInt = 8


class AgentGold(FrozenModel):
    """Evaluator/trainer-only gold. Never included in model input records."""

    schema_version: Literal["finance_agent.gold.v1"] = "finance_agent.gold.v1"
    trajectory: AgentTrajectory
    expected_output: ExpectedAgentOutput
    oracle: OracleMeta
    gold_sha256: Sha256Hex

    @model_validator(mode="after")
    def _hash_matches(self, info: ValidationInfo) -> Self:
        if not info.context or not info.context.get("skip_hash_validation", False):
            if self.gold_sha256 != content_sha256(self.canonical_payload()):
                raise ValueError("gold_sha256 mismatch")
        final = self.trajectory.turns[-1].final_answer
        if final != self.expected_output.final_answer:
            raise ValueError("trajectory final_answer must equal expected_output.final_answer")
        return self

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"gold_sha256"})

    @classmethod
    def seal(
        cls,
        *,
        trajectory: AgentTrajectory,
        expected_output: ExpectedAgentOutput,
        oracle: OracleMeta,
    ) -> AgentGold:
        provisional = cls.model_validate(
            {
                "trajectory": trajectory,
                "expected_output": expected_output,
                "oracle": oracle,
                "gold_sha256": "0" * 64,
            },
            context={"skip_hash_validation": True},
        )
        payload = provisional.canonical_payload()
        return cls.model_validate({**payload, "gold_sha256": content_sha256(payload)})


class AgentProvenance(FrozenModel):
    split: SplitName
    case_family: CaseFamily
    template_family: BoundedStr
    label_source: LabelSource
    scenario_fingerprint: Sha256Hex
    held_out_tools: tuple[ToolName, ...] = ()
    held_out_domains: tuple[BoundedStr, ...] = ()


class EconomicsObservation(FrozenModel):
    """Measured inference economics only; generated corpora stay unmeasured/null."""

    source: Literal["unmeasured", "measured"] = "unmeasured"
    latency_ms: NonNegativeSafeInt | None = None
    cost_usd_micros: NonNegativeSafeInt | None = None

    @model_validator(mode="after")
    def _no_synthetic_economics(self) -> Self:
        if self.source == "unmeasured" and (
            self.latency_ms is not None or self.cost_usd_micros is not None
        ):
            raise ValueError("unmeasured economics must be null")
        if self.source == "measured" and self.latency_ms is None:
            raise ValueError("measured economics requires observed latency_ms")
        return self


class AgentEpisodeEnvelope(FrozenModel):
    """Canonical envelope with explicit model-input and evaluator-only gold boundaries."""

    schema_version: Literal["finance_agent.v1"] = SCHEMA_VERSION_FINANCE_AGENT
    example_id: ExampleId
    world_id: WorldId
    group_id: GroupId
    difficulty: Difficulty
    case_family: CaseFamily
    model_input: AgentModelInput
    gold: AgentGold
    provenance: AgentProvenance
    economics: EconomicsObservation = Field(default_factory=EconomicsObservation)
    episode_sha256: Sha256Hex

    @model_validator(mode="after")
    def _invariants(self, info: ValidationInfo) -> Self:
        if self.example_id != self.model_input.example_id:
            raise ValueError("example_id must match model_input.example_id")
        if self.case_family != self.provenance.case_family:
            raise ValueError("case_family must match provenance.case_family")
        if self.provenance.label_source != self.gold.oracle.label_source:
            raise ValueError("provenance label_source must match oracle metadata")
        input_users = tuple(
            (message.turn_index, message.text) for message in self.model_input.user_messages
        )
        trajectory_users = tuple(
            (turn.turn_index, turn.text)
            for turn in self.gold.trajectory.turns
            if turn.role is TurnRole.USER
        )
        if input_users != trajectory_users:
            raise ValueError("model_input user_messages must match trajectory user turns")
        available = {tool.name for tool in self.model_input.tools}
        for call in self.gold.trajectory.tool_calls():
            if call.tool not in available:
                raise ValueError(f"gold tool {call.tool} is absent from model_input tools")
        if len(self.gold.trajectory.tool_calls()) > self.gold.expected_output.max_tool_calls:
            raise ValueError("trajectory exceeds expected max_tool_calls")
        if not info.context or not info.context.get("skip_hash_validation", False):
            if self.episode_sha256 != content_sha256(self.canonical_payload()):
                raise ValueError("episode_sha256 mismatch")
        return self

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"episode_sha256"})

    @classmethod
    def seal(cls, **data: Any) -> AgentEpisodeEnvelope:
        provisional = cls.model_validate(
            {**data, "episode_sha256": "0" * 64},
            context={"skip_hash_validation": True},
        )
        payload = provisional.canonical_payload()
        return cls.model_validate({**payload, "episode_sha256": content_sha256(payload)})

    def model_record(self) -> dict[str, Any]:
        """Input-only row safe for inference and evaluator/model processes."""
        return self.model_input.model_dump(mode="json")

    def gold_record(self) -> dict[str, Any]:
        """Gold row for isolated trainer/evaluator access."""
        return {
            "schema_version": "finance_agent.gold_record.v1",
            "example_id": self.example_id,
            "world_id": self.world_id,
            "group_id": self.group_id,
            "difficulty": self.difficulty.value,
            "case_family": self.case_family.value,
            "model_input_sha256": self.model_input.model_input_sha256,
            "gold": self.gold.model_dump(mode="json"),
            "provenance": self.provenance.model_dump(mode="json"),
            "economics": self.economics.model_dump(mode="json"),
            "episode_sha256": self.episode_sha256,
        }


class MoneyMinor(FrozenModel):
    amount_minor: SafeInt
    currency: Literal["USD"] = "USD"


__all__ = [
    "SCHEMA_VERSION_FINANCE_AGENT",
    "TECHNIQUE_ID_AGENT_TRAJECTORY",
    "AgentEpisodeEnvelope",
    "AgentGold",
    "AgentModelInput",
    "AgentProvenance",
    "AgentTrajectory",
    "CaseFamily",
    "EconomicsObservation",
    "ExpectedAgentOutput",
    "FinalAnswer",
    "FinalAnswerKind",
    "MoneyMinor",
    "OracleMeta",
    "ProvenanceRef",
    "ResultBinding",
    "TeacherRolloutEvidence",
    "ToolCall",
    "ToolDefinition",
    "ToolName",
    "ToolResult",
    "TrajectoryTurn",
    "TurnRole",
    "UserMessage",
]
