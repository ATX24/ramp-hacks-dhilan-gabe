"""Runtime configuration for the inference serving plane."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(
    env: dict[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _env_float(
    env: dict[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _env_bool(env: dict[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    """Bounded, deterministic defaults for demo QPS."""

    model_bundle_root: Path
    endpoint_id: str
    max_prompt_tokens: int
    max_completion_tokens: int
    max_input_bytes: int
    request_timeout_s: float
    max_concurrent_requests: int
    temperature: float
    top_p: float
    seed: int
    require_offline: bool
    runtime_backend: str

    @classmethod
    def from_environ(cls, environ: dict[str, str] | None = None) -> InferenceSettings:
        env = environ if environ is not None else dict(os.environ)
        root = Path(
            env.get(
                "DISTILLERY_INFERENCE_MODEL_ROOT",
                env.get("SM_MODEL_DIR", "/opt/ml/model"),
            )
        )
        return cls(
            model_bundle_root=root,
            endpoint_id=env.get(
                "DISTILLERY_INFERENCE_ENDPOINT_ID",
                "distillery-demo-inference",
            ),
            max_prompt_tokens=_env_int(
                env,
                "DISTILLERY_INFERENCE_MAX_PROMPT_TOKENS",
                1024,
                minimum=64,
                maximum=4096,
            ),
            max_completion_tokens=_env_int(
                env,
                "DISTILLERY_INFERENCE_MAX_COMPLETION_TOKENS",
                192,
                minimum=16,
                maximum=2048,
            ),
            max_input_bytes=_env_int(
                env,
                "DISTILLERY_INFERENCE_MAX_INPUT_BYTES",
                65536,
                minimum=1024,
                maximum=1_048_576,
            ),
            request_timeout_s=_env_float(
                env,
                "DISTILLERY_INFERENCE_TIMEOUT_S",
                25.0,
                minimum=1.0,
                maximum=120.0,
            ),
            max_concurrent_requests=_env_int(
                env,
                "DISTILLERY_INFERENCE_MAX_CONCURRENT",
                4,
                minimum=1,
                maximum=32,
            ),
            temperature=_env_float(
                env,
                "DISTILLERY_INFERENCE_TEMPERATURE",
                0.0,
                minimum=0.0,
                maximum=1.0,
            ),
            top_p=_env_float(
                env,
                "DISTILLERY_INFERENCE_TOP_P",
                1.0,
                minimum=0.0,
                maximum=1.0,
            ),
            seed=_env_int(
                env,
                "DISTILLERY_INFERENCE_SEED",
                17,
                minimum=0,
                maximum=2_147_483_647,
            ),
            require_offline=_env_bool(env, "DISTILLERY_INFERENCE_REQUIRE_OFFLINE", True),
            runtime_backend=env.get("DISTILLERY_INFERENCE_RUNTIME", "torch").strip().lower(),
        )


def enforce_offline_environment(environ: dict[str, str] | None = None) -> None:
    """Fail loud unless Hugging Face / Transformers offline mode is enforced."""
    env = environ if environ is not None else dict(os.environ)
    required = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    missing = [
        key
        for key, expected in required.items()
        if env.get(key, "").strip() not in {expected, "true", "True"}
    ]
    if missing:
        raise RuntimeError(
            "offline/network isolation is required for inference serving; "
            f"missing or unset: {', '.join(missing)}"
        )
