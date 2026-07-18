"""Resume / stop / status helpers for emergency SageMaker jobs (injectable client)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from experiments.aws_smoke.profile import RunArm


class SageMakerClient(Protocol):
    def describe_training_job(self, *, TrainingJobName: str) -> dict[str, Any]: ...

    def stop_training_job(self, *, TrainingJobName: str) -> dict[str, Any]: ...

    def list_training_jobs(self, **kwargs: Any) -> dict[str, Any]: ...


TERMINAL = frozenset({"Completed", "Failed", "Stopped"})
ACTIVE = frozenset({"InProgress", "Stopping"})


@dataclass(frozen=True, slots=True)
class JobStatusView:
    job_name: str
    arm: RunArm | None
    status: str
    secondary_status: str | None
    failure_reason: str | None
    terminal: bool
    exists: bool


class JobNotFoundError(LookupError):
    pass


def _is_not_found(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if code in {"ValidationException", "ResourceNotFound"}:
            return True
    message = str(exc).lower()
    return "could not find" in message or "not found" in message


def status_for_job(
    client: SageMakerClient,
    job_name: str,
    *,
    arm: RunArm | None = None,
    allow_missing: bool = False,
) -> JobStatusView:
    try:
        raw = client.describe_training_job(TrainingJobName=job_name)
    except Exception as exc:  # noqa: BLE001 - normalize boto + mock failures
        if allow_missing and _is_not_found(exc):
            return JobStatusView(
                job_name=job_name,
                arm=arm,
                status="NotStarted",
                secondary_status=None,
                failure_reason=None,
                terminal=False,
                exists=False,
            )
        raise JobNotFoundError(str(exc)) from exc
    status = str(raw.get("TrainingJobStatus", "Unknown"))
    return JobStatusView(
        job_name=job_name,
        arm=arm,
        status=status,
        secondary_status=raw.get("SecondaryStatus"),
        failure_reason=raw.get("FailureReason"),
        terminal=status in TERMINAL,
        exists=True,
    )


def stop_job(client: SageMakerClient, job_name: str) -> JobStatusView:
    current = status_for_job(client, job_name, allow_missing=False)
    if current.terminal or current.status == "Stopping":
        return current
    client.stop_training_job(TrainingJobName=job_name)
    return status_for_job(client, job_name, allow_missing=False)


def status_campaign(
    client: SageMakerClient,
    jobs: Mapping[RunArm, str],
) -> dict[RunArm, JobStatusView]:
    return {
        arm: status_for_job(client, job_name, arm=arm, allow_missing=True)
        for arm, job_name in jobs.items()
    }


def next_resumable_arm(
    statuses: Mapping[RunArm, JobStatusView],
    *,
    ordered_arms: Sequence[RunArm],
) -> RunArm | None:
    """
    Serial resume: return the first arm that is not terminal and not active.

    If any job is still active, return None (quota=1; wait).
    """
    if any(view.status in ACTIVE for view in statuses.values()):
        return None
    for arm in ordered_arms:
        view = statuses.get(arm)
        if view is None or not view.exists:
            return arm
        if not view.terminal:
            return arm
    return None


def inventory_smoke_jobs(
    client: SageMakerClient,
    *,
    name_contains: str = "aws-smoke-",
) -> list[str]:
    names: set[str] = set()
    for status in ("InProgress", "Stopping", "Completed", "Failed", "Stopped"):
        response = client.list_training_jobs(
            StatusEquals=status,
            NameContains=name_contains,
            MaxResults=100,
        )
        for summary in response.get("TrainingJobSummaries", []):
            names.add(str(summary["TrainingJobName"]))
    return sorted(names)
