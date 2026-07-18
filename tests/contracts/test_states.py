"""Monotonic DistillationRun state machine tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from distillery.contracts.errors import DistilleryErrorCode
from distillery.contracts.recipes import (
    AUTO_BASELINE_PRECEDENCE_REASON,
    AUTO_SEQUENCE_RESPONSES_REASONS,
)
from distillery.contracts.run import DistillationRun, RunFailure
from distillery.contracts.states import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    InvalidTransitionError,
    RunState,
    assert_transition,
    next_states,
)

HEX64 = "a" * 64
ARTIFACT_ID = "art_tinyfable_001"
REPORT_ID = "prf_proof_001"


def _ts() -> datetime:
    return datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)


def _queued() -> DistillationRun:
    return DistillationRun(
        run_id="run_state_001",
        dataset_id="ds_finance_world_v1",
        state=RunState.QUEUED,
        manifest_sha256=HEX64,
        requested_recipe="auto",
        resolved_recipe="sequence.v1",
        resolver_reasons=AUTO_SEQUENCE_RESPONSES_REASONS,
        created_at=_ts(),
        updated_at=_ts(),
    )


def _unresolved() -> DistillationRun:
    payload = _queued().model_dump(mode="python")
    payload["resolved_recipe"] = None
    payload["resolver_reasons"] = ()
    return DistillationRun.model_validate(payload)


def test_happy_path_with_synthesizing() -> None:
    run = _queued()
    path = [
        RunState.STARTING,
        RunState.SYNTHESIZING,
        RunState.TRAINING,
        RunState.EVALUATING,
        RunState.FINALIZING,
        RunState.SUCCEEDED,
    ]
    for target in path:
        if target is RunState.SUCCEEDED:
            run = run.transition(
                target,
                at=_ts(),
                model_artifact_id=ARTIFACT_ID,
                proof_report_id=REPORT_ID,
            )
        else:
            run = run.transition(target, at=_ts())
    assert run.state is RunState.SUCCEEDED
    assert run.is_terminal()
    assert len(run.transitions) == len(path)


def test_skip_synthesizing_records_reason() -> None:
    run = _queued().transition(RunState.STARTING, at=_ts())
    run = run.transition(
        RunState.TRAINING,
        at=_ts(),
        skip_synthesis_reason="responses_already_present",
    )
    assert run.state is RunState.TRAINING
    assert run.skip_synthesis_reason == "responses_already_present"
    assert run.transitions[-1].skip_synthesis_reason == "responses_already_present"


def test_starting_to_training_cannot_skip_silently() -> None:
    run = _queued().transition(RunState.STARTING, at=_ts())
    with pytest.raises(InvalidTransitionError, match="synthesis skip audit"):
        run.transition(RunState.TRAINING, at=_ts())


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RunState.QUEUED, RunState.TRAINING),
        (RunState.STARTING, RunState.EVALUATING),
        (RunState.TRAINING, RunState.STARTING),
        (RunState.EVALUATING, RunState.TRAINING),
        (RunState.SUCCEEDED, RunState.FAILED),
        (RunState.FAILED, RunState.QUEUED),
        (RunState.CANCELLED, RunState.STARTING),
        (RunState.FINALIZING, RunState.TRAINING),
        (RunState.FINALIZING, RunState.CANCELLED),
    ],
)
def test_invalid_transitions_rejected(current: RunState, target: RunState) -> None:
    with pytest.raises(InvalidTransitionError) as exc:
        assert_transition(current, target)
    assert exc.value.code is DistilleryErrorCode.INVALID_TRANSITION
    assert not exc.value.payload.retryable


def test_all_non_allowed_edges_fail() -> None:
    for current in RunState:
        allowed = ALLOWED_TRANSITIONS[current]
        for target in RunState:
            if target in allowed:
                assert_transition(current, target)
            else:
                with pytest.raises(InvalidTransitionError):
                    assert_transition(current, target)


def test_terminal_states_have_no_exits() -> None:
    for terminal in TERMINAL_STATES:
        assert next_states(terminal) == frozenset()
        run = _queued()
        # Drive to terminal via cancel from QUEUED.
        if terminal is RunState.CANCELLED:
            run = run.transition(RunState.CANCELLED, at=_ts())
        elif terminal is RunState.FAILED:
            run = run.transition(
                RunState.FAILED,
                at=_ts(),
                failure=RunFailure(
                    code=DistilleryErrorCode.AWS_JOB_FAILED,
                    message="job failed",
                ),
            )
        else:
            for step in (
                RunState.STARTING,
                RunState.SYNTHESIZING,
                RunState.TRAINING,
                RunState.EVALUATING,
                RunState.FINALIZING,
                RunState.SUCCEEDED,
            ):
                if step is RunState.SUCCEEDED:
                    run = run.transition(
                        step,
                        at=_ts(),
                        model_artifact_id=ARTIFACT_ID,
                        proof_report_id=REPORT_ID,
                    )
                else:
                    run = run.transition(step, at=_ts())
        assert run.is_terminal()
        with pytest.raises(InvalidTransitionError):
            run.transition(RunState.QUEUED, at=_ts())


def test_failed_transition_requires_typed_failure() -> None:
    with pytest.raises(ValueError, match="requires a typed failure"):
        _queued().transition(RunState.FAILED, at=_ts())


def test_success_transition_requires_both_output_references() -> None:
    run = _queued()
    for step in (
        RunState.STARTING,
        RunState.SYNTHESIZING,
        RunState.TRAINING,
        RunState.EVALUATING,
        RunState.FINALIZING,
    ):
        run = run.transition(step, at=_ts())
    with pytest.raises(ValueError, match="requires model_artifact_id"):
        run.transition(RunState.SUCCEEDED, at=_ts())


def test_terminal_payloads_validated_on_direct_construction() -> None:
    payload = _queued().model_dump(mode="python")
    payload["state"] = RunState.FAILED
    with pytest.raises(ValidationError, match="require a typed failure"):
        DistillationRun.model_validate(payload)
    payload["failure"] = RunFailure(
        code=DistilleryErrorCode.AWS_JOB_FAILED,
        message="failed",
    )
    with pytest.raises(ValidationError, match="require transition history"):
        DistillationRun.model_validate(payload)


def test_finalization_wins_cancellation_race() -> None:
    run = _queued()
    for step in (
        RunState.STARTING,
        RunState.SYNTHESIZING,
        RunState.TRAINING,
        RunState.EVALUATING,
        RunState.FINALIZING,
    ):
        run = run.transition(step, at=_ts())
    with pytest.raises(InvalidTransitionError):
        run.transition(RunState.CANCELLED, at=_ts())
    completed = run.transition(
        RunState.SUCCEEDED,
        at=_ts(),
        model_artifact_id=ARTIFACT_ID,
        proof_report_id=REPORT_ID,
    )
    assert completed.state is RunState.SUCCEEDED


def test_run_resource_also_rejects_silent_recipe_downgrade() -> None:
    payload = _queued().model_dump(mode="python")
    payload.update(
        {
            "requested_recipe": "logit.v1",
            "resolved_recipe": "sequence.v1",
            "resolver_reasons": ("explicit_request",),
        }
    )
    with pytest.raises(ValidationError, match="must resolve only to"):
        DistillationRun.model_validate(payload)


def test_run_transition_can_audit_recipe_resolution() -> None:
    run = _unresolved().transition(
        RunState.STARTING,
        at=_ts(),
        resolved_recipe="sequence.v1",
        resolver_reasons=AUTO_SEQUENCE_RESPONSES_REASONS,
    )
    assert run.resolved_recipe == "sequence.v1"
    assert run.resolver_reasons == AUTO_SEQUENCE_RESPONSES_REASONS


def test_trainable_run_cannot_use_direct_success_edge() -> None:
    with pytest.raises(InvalidTransitionError, match="reserved for do_not_distill"):
        _queued().transition(
            RunState.SUCCEEDED,
            at=_ts(),
            model_artifact_id=ARTIFACT_ID,
            proof_report_id=REPORT_ID,
        )


def test_do_not_distill_uses_direct_terminal_report_path() -> None:
    completed = _unresolved().transition(
        RunState.SUCCEEDED,
        at=_ts(),
        resolved_recipe="do_not_distill",
        resolver_reasons=(AUTO_BASELINE_PRECEDENCE_REASON,),
        proof_report_id=REPORT_ID,
    )
    assert completed.is_terminal()
    assert completed.proof_report_id == REPORT_ID
    assert completed.model_artifact_id is None
    assert len(completed.transitions) == 1


@pytest.mark.parametrize(
    "target",
    [RunState.STARTING, RunState.SYNTHESIZING, RunState.TRAINING],
)
def test_do_not_distill_never_enters_execution_states(target: RunState) -> None:
    payload = _unresolved().model_dump(mode="python")
    payload.update(
        {
            "resolved_recipe": "do_not_distill",
            "resolver_reasons": (AUTO_BASELINE_PRECEDENCE_REASON,),
        }
    )
    run = DistillationRun.model_validate(payload)
    if target is RunState.STARTING:
        with pytest.raises(InvalidTransitionError, match="terminal report path"):
            run.transition(target, at=_ts())
        return

    payload["state"] = target
    payload["transitions"] = (
        {
            "from_state": "QUEUED",
            "to_state": target,
            "at": _ts(),
        },
    )
    with pytest.raises(ValidationError, match="cannot enter execution states"):
        DistillationRun.model_validate(payload)
