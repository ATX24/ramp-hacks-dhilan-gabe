"""Asynchronous DistillationRun resource and transition helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import (
    Field,
    StrictStr,
    model_validator,
)

from distillery.contracts.base import FrozenJsonObject, FrozenModel
from distillery.contracts.errors import DistilleryErrorCode, ErrorPayload
from distillery.contracts.hashing import AwareDatetime, Sha256Hex, content_sha256
from distillery.contracts.ids import ArtifactId, DatasetId, ProofReportId, RunId
from distillery.contracts.recipes import (
    RequestedRecipe,
    ResolvedRecipe,
    validate_recipe_resolution,
)
from distillery.contracts.states import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    InvalidTransitionError,
    RunState,
    assert_transition,
)

SkipSynthesisReason = Literal["responses_already_present"]


class TransitionRecord(FrozenModel):
    from_state: RunState
    to_state: RunState
    at: AwareDatetime
    reason: StrictStr | None = None
    skip_synthesis_reason: SkipSynthesisReason | None = None


class RunFailure(FrozenModel):
    code: DistilleryErrorCode
    message: StrictStr = Field(min_length=1)
    details: FrozenJsonObject = Field(default_factory=dict)


class DistillationRun(FrozenModel):
    """Asynchronous DistillationRun with monotonic state history."""

    schema_version: Literal["distillery.run_resource.v1"] = "distillery.run_resource.v1"
    run_id: RunId
    dataset_id: DatasetId
    state: RunState
    manifest_sha256: Sha256Hex
    requested_recipe: RequestedRecipe
    resolved_recipe: ResolvedRecipe | None = None
    resolver_reasons: tuple[StrictStr, ...] = ()
    backend_job_ref: StrictStr | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    transitions: tuple[TransitionRecord, ...] = ()
    skip_synthesis_reason: SkipSynthesisReason | None = None
    failure: RunFailure | None = None
    model_artifact_id: ArtifactId | None = None
    proof_report_id: ProofReportId | None = None

    @model_validator(mode="after")
    def _validate_resource_invariants(self) -> DistillationRun:
        for name, value in (("created_at", self.created_at), ("updated_at", self.updated_at)):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")

        if self.resolved_recipe is None:
            if self.resolver_reasons:
                raise ValueError("resolver_reasons require resolved_recipe")
        else:
            validate_recipe_resolution(
                self.requested_recipe,
                self.resolved_recipe,
                self.resolver_reasons,
            )
        if self.state in {
            RunState.STARTING,
            RunState.SYNTHESIZING,
            RunState.TRAINING,
            RunState.EVALUATING,
            RunState.FINALIZING,
            RunState.SUCCEEDED,
        } and self.resolved_recipe is None:
            raise ValueError("runs must resolve a recipe before leaving QUEUED")

        if self.state is RunState.FAILED:
            if self.failure is None:
                raise ValueError("FAILED runs require a typed failure")
        elif self.failure is not None:
            raise ValueError("only FAILED runs may carry a failure")

        has_artifact = self.model_artifact_id is not None
        has_report = self.proof_report_id is not None
        if self.state is RunState.SUCCEEDED:
            if self.resolved_recipe == "do_not_distill":
                if has_artifact or not has_report:
                    raise ValueError(
                        "do_not_distill success requires only proof_report_id"
                    )
            elif not (has_artifact and has_report):
                raise ValueError(
                    "trainable SUCCEEDED runs require model_artifact_id "
                    "and proof_report_id"
                )
        elif has_artifact or has_report:
            raise ValueError("output resource references are valid only on SUCCEEDED runs")

        if self.resolved_recipe == "do_not_distill":
            if self.state not in {
                RunState.QUEUED,
                RunState.SUCCEEDED,
                RunState.FAILED,
                RunState.CANCELLED,
            }:
                raise ValueError(
                    "do_not_distill runs cannot enter execution states"
                )
            if self.backend_job_ref is not None:
                raise ValueError("do_not_distill runs cannot carry a backend job")
            if self.skip_synthesis_reason is not None:
                raise ValueError("do_not_distill runs cannot carry a synthesis skip")
            if self.transitions and (
                len(self.transitions) != 1
                or self.transitions[0].from_state is not RunState.QUEUED
                or self.transitions[0].to_state
                not in {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED}
            ):
                raise ValueError(
                    "do_not_distill may only leave QUEUED for a terminal state"
                )

        expected_from = RunState.QUEUED
        previous_at = self.created_at
        skip_recorded = False
        for record in self.transitions:
            if record.from_state is not expected_from:
                raise ValueError("transition history is not contiguous")
            if record.to_state not in ALLOWED_TRANSITIONS[record.from_state]:
                raise ValueError(
                    f"transition history contains invalid edge "
                    f"{record.from_state.value} -> {record.to_state.value}"
                )
            if record.at < previous_at:
                raise ValueError("transition timestamps must be monotonic")
            if record.at > self.updated_at:
                raise ValueError("transition timestamp cannot exceed updated_at")
            is_skip = (
                record.from_state is RunState.STARTING
                and record.to_state is RunState.TRAINING
            )
            if is_skip:
                if record.skip_synthesis_reason != "responses_already_present":
                    raise ValueError(
                        "STARTING -> TRAINING requires "
                        "skip_synthesis_reason='responses_already_present'"
                    )
                skip_recorded = True
            elif record.skip_synthesis_reason is not None:
                raise ValueError("skip_synthesis_reason is valid only on STARTING -> TRAINING")
            expected_from = record.to_state
            previous_at = record.at

        if self.transitions:
            if expected_from is not self.state:
                raise ValueError("state must equal the final transition target")
        elif self.state is not RunState.QUEUED:
            raise ValueError("non-QUEUED runs require transition history")

        pending_skip = self.state in {RunState.QUEUED, RunState.STARTING}
        if skip_recorded:
            if self.skip_synthesis_reason != "responses_already_present":
                raise ValueError("run must retain the audited synthesis skip reason")
        elif self.skip_synthesis_reason is not None and not pending_skip:
            raise ValueError("synthesis skip reason was never used by a transition")
        return self

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def transition(
        self,
        target: RunState,
        *,
        at: AwareDatetime,
        reason: StrictStr | None = None,
        failure: RunFailure | None = None,
        model_artifact_id: ArtifactId | None = None,
        proof_report_id: ProofReportId | None = None,
        resolved_recipe: ResolvedRecipe | None = None,
        resolver_reasons: tuple[StrictStr, ...] | None = None,
        backend_job_ref: StrictStr | None = None,
        skip_synthesis_reason: SkipSynthesisReason | None = None,
    ) -> DistillationRun:
        """Return a new run advanced to ``target`` (immutable update)."""
        assert_transition(self.state, target)
        if not isinstance(at, datetime):
            raise ValueError("transition timestamp must be a datetime")
        if at.tzinfo is None or at.utcoffset() is None:
            raise ValueError("transition timestamp must be timezone-aware")
        if at < self.updated_at:
            raise ValueError("transition timestamp cannot precede updated_at")

        effective_resolved = resolved_recipe or self.resolved_recipe
        if target in {
            RunState.STARTING,
            RunState.SYNTHESIZING,
            RunState.TRAINING,
            RunState.EVALUATING,
            RunState.FINALIZING,
            RunState.SUCCEEDED,
        } and effective_resolved is None:
            raise InvalidTransitionError(
                self.state,
                target,
                reason="recipe must be resolved before execution",
            )
        if effective_resolved == "do_not_distill" and target not in {
            RunState.SUCCEEDED,
            RunState.FAILED,
            RunState.CANCELLED,
        }:
            raise InvalidTransitionError(
                self.state,
                target,
                reason="do_not_distill uses the direct terminal report path",
            )
        if (
            self.state is RunState.QUEUED
            and target is RunState.SUCCEEDED
            and effective_resolved != "do_not_distill"
        ):
            raise InvalidTransitionError(
                self.state,
                target,
                reason="direct success is reserved for do_not_distill",
            )

        effective_skip_reason: SkipSynthesisReason | None = None
        if self.state is RunState.STARTING and target is RunState.TRAINING:
            effective_skip_reason = skip_synthesis_reason or self.skip_synthesis_reason
            if effective_skip_reason != "responses_already_present":
                raise InvalidTransitionError(
                    self.state,
                    target,
                    reason="missing responses_already_present synthesis skip audit",
                )
        elif skip_synthesis_reason is not None:
            raise ValueError("skip_synthesis_reason is valid only on STARTING -> TRAINING")

        if target is RunState.FAILED and failure is None:
            raise ValueError("transition to FAILED requires a typed failure")
        if target is not RunState.FAILED and failure is not None:
            raise ValueError("failure payload is valid only for FAILED")
        if target is RunState.SUCCEEDED:
            if effective_resolved == "do_not_distill":
                if model_artifact_id is not None or proof_report_id is None:
                    raise ValueError(
                        "do_not_distill success requires proof_report_id "
                        "and forbids model_artifact_id"
                    )
            elif model_artifact_id is None or proof_report_id is None:
                raise ValueError(
                    "trainable success requires model_artifact_id and proof_report_id"
                )
        elif model_artifact_id is not None or proof_report_id is not None:
            raise ValueError("output resource references are valid only for SUCCEEDED")

        record = TransitionRecord(
            from_state=self.state,
            to_state=target,
            at=at,
            reason=reason,
            skip_synthesis_reason=effective_skip_reason,
        )
        updates: dict[str, object] = {
            "state": target,
            "updated_at": at,
            "transitions": (*self.transitions, record),
        }
        if failure is not None:
            updates["failure"] = failure
        if model_artifact_id is not None:
            updates["model_artifact_id"] = model_artifact_id
        if proof_report_id is not None:
            updates["proof_report_id"] = proof_report_id
        if resolved_recipe is not None:
            updates["resolved_recipe"] = resolved_recipe
        if resolver_reasons is not None:
            updates["resolver_reasons"] = resolver_reasons
        if backend_job_ref is not None:
            updates["backend_job_ref"] = backend_job_ref
        if effective_skip_reason is not None:
            updates["skip_synthesis_reason"] = effective_skip_reason
        payload = self.model_dump(mode="python")
        payload.update(updates)
        return type(self).model_validate(payload)

    def resource_hash(self) -> str:
        payload = self.model_dump(mode="python", exclude={"updated_at"})
        return content_sha256(payload)

    def failure_payload(self) -> ErrorPayload | None:
        if self.failure is None:
            return None
        return ErrorPayload.from_code(
            self.failure.code,
            self.failure.message,
            details=self.failure.details,
            run_id=self.run_id,
        )
