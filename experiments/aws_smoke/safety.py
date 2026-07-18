"""Launch safety gates: non-root identity, gabriel-cli profile, explicit confirm."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from experiments.aws_smoke.pins import EmergencyEvidence

CONFIRM_PHRASE = "I_CONFIRM_SAGEMAKER_SUBMIT"
REQUIRED_PROFILE = "gabriel-cli"
_ROOT_ARN_RE = re.compile(r":root$|:root/")
_ROOT_USER_RE = re.compile(r"arn:aws:iam::[0-9]{12}:root$")


@dataclass(frozen=True, slots=True)
class CallerIdentity:
    account: str
    arn: str
    user_id: str


@dataclass(frozen=True, slots=True)
class SafetyGateResult:
    ok: bool
    profile: str
    identity: CallerIdentity | None
    violations: tuple[str, ...]


def assert_confirmation(confirm: str | None) -> None:
    if confirm != CONFIRM_PHRASE:
        raise PermissionError(
            f"refusing launch without exact confirmation phrase {CONFIRM_PHRASE!r}"
        )


def assert_profile(profile: str | None, *, evidence: EmergencyEvidence) -> None:
    if profile != REQUIRED_PROFILE:
        raise PermissionError(
            f"refusing launch: aws profile must be {REQUIRED_PROFILE!r}, got {profile!r}"
        )
    if evidence.aws_profile != REQUIRED_PROFILE:
        raise PermissionError("evidence.aws_profile must be gabriel-cli")


def is_root_identity(identity: CallerIdentity | Mapping[str, Any]) -> bool:
    if isinstance(identity, CallerIdentity):
        arn = identity.arn
    else:
        arn = str(identity.get("Arn", ""))
    return bool(_ROOT_ARN_RE.search(arn) or _ROOT_USER_RE.fullmatch(arn))


def assert_non_root(identity: CallerIdentity) -> None:
    if is_root_identity(identity):
        raise PermissionError(
            f"refusing launch: caller is root identity ({identity.arn}). "
            "Configure a non-root gabriel-cli principal first."
        )


def evaluate_safety_gates(
    *,
    profile: str | None,
    confirm: str | None,
    evidence: EmergencyEvidence,
    identity: CallerIdentity | None,
    dry_run: bool,
) -> SafetyGateResult:
    violations: list[str] = []
    if profile != REQUIRED_PROFILE:
        violations.append(f"profile_must_be_{REQUIRED_PROFILE}")
    if evidence.aws_profile != REQUIRED_PROFILE:
        violations.append("evidence_profile_mismatch")
    if identity is None and not dry_run:
        violations.append("missing_caller_identity")
    if identity is not None:
        if identity.account != evidence.aws_account_id:
            violations.append("account_mismatch")
        if is_root_identity(identity):
            violations.append("root_identity_forbidden")
    if not dry_run and confirm != CONFIRM_PHRASE:
        violations.append("missing_confirmation_phrase")
    return SafetyGateResult(
        ok=not violations,
        profile=profile or "",
        identity=identity,
        violations=tuple(violations),
    )


def enforce_safety_gates(
    *,
    profile: str | None,
    confirm: str | None,
    evidence: EmergencyEvidence,
    identity_provider: Callable[[], CallerIdentity] | None,
    dry_run: bool,
) -> CallerIdentity | None:
    """Fail loud before any CreateTrainingJob mutation (dry-run may skip STS)."""
    identity: CallerIdentity | None = None
    if identity_provider is not None and (not dry_run or profile == REQUIRED_PROFILE):
        # Dry-run still resolves identity when a provider is supplied so operators
        # see root/account failures before spending time on uploads.
        try:
            identity = identity_provider()
        except Exception as exc:  # noqa: BLE001 - surface STS failures explicitly
            if not dry_run:
                raise PermissionError(f"failed to resolve caller identity: {exc}") from exc
            identity = None
    result = evaluate_safety_gates(
        profile=profile,
        confirm=confirm,
        evidence=evidence,
        identity=identity,
        dry_run=dry_run,
    )
    if not result.ok:
        raise PermissionError(
            "aws_smoke safety gates failed: " + ", ".join(result.violations)
        )
    if not dry_run:
        assert_confirmation(confirm)
        assert_profile(profile, evidence=evidence)
        if identity is None:
            raise PermissionError("caller identity required for non-dry-run launch")
        assert_non_root(identity)
    return identity
