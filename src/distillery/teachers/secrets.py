"""Runtime-only Anthropic credential resolution.

Secret values are wrapped in ``SecretStr`` and passed only to a client factory.
They are never fields on teacher requests, policies, provenance, or caches.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Final, Protocol

from pydantic import SecretStr

from distillery.teachers.errors import TeacherErrorCode, raise_teacher_error

ANTHROPIC_SECRET_NAME: Final = "ANTHROPIC_API_KEY"


class AnthropicSecretResolver(Protocol):
    def resolve(self, name: str) -> SecretStr | None:
        """Resolve a named secret without logging or persisting it."""


class EnvironmentAnthropicSecretResolver:
    """Read only ``ANTHROPIC_API_KEY`` from the process environment."""

    def resolve(self, name: str) -> SecretStr | None:
        _require_supported_name(name)
        value = os.environ.get(ANTHROPIC_SECRET_NAME)
        return SecretStr(value) if value else None


class MacOSKeychainAnthropicSecretResolver:
    """Read a generic-password item whose service is ``ANTHROPIC_API_KEY``."""

    def resolve(self, name: str) -> SecretStr | None:
        _require_supported_name(name)
        if sys.platform != "darwin":
            return None
        try:
            completed = subprocess.run(
                [
                    "/usr/bin/security",
                    "find-generic-password",
                    "-s",
                    ANTHROPIC_SECRET_NAME,
                    "-w",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        value = completed.stdout.rstrip("\r\n")
        return SecretStr(value) if value else None


class ChainedAnthropicSecretResolver:
    def __init__(self, *resolvers: AnthropicSecretResolver) -> None:
        if not resolvers:
            raise ValueError("at least one secret resolver is required")
        self._resolvers = resolvers

    def resolve(self, name: str) -> SecretStr | None:
        _require_supported_name(name)
        for resolver in self._resolvers:
            secret = resolver.resolve(name)
            if secret is not None:
                return secret
        return None


def require_anthropic_secret(resolver: AnthropicSecretResolver) -> SecretStr:
    secret = resolver.resolve(ANTHROPIC_SECRET_NAME)
    if secret is None:
        raise_teacher_error(
            TeacherErrorCode.SECRET_UNAVAILABLE,
            "Anthropic credential is unavailable from the injected secret resolver",
            details={"secret_name": ANTHROPIC_SECRET_NAME},
        )
    return secret


def _require_supported_name(name: str) -> None:
    if name != ANTHROPIC_SECRET_NAME:
        raise ValueError("Anthropic resolver may read only ANTHROPIC_API_KEY")


__all__ = [
    "ANTHROPIC_SECRET_NAME",
    "AnthropicSecretResolver",
    "ChainedAnthropicSecretResolver",
    "EnvironmentAnthropicSecretResolver",
    "MacOSKeychainAnthropicSecretResolver",
    "require_anthropic_secret",
]
