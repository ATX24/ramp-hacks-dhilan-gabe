"""Deterministic sealed model-bundle fixtures for inference tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from distillery_inference.bundle import LoadedBundle, load_serving_bundle
from distillery_inference.config import InferenceSettings
from distillery_inference.runtime import FakeRuntime
from distillery_inference.service import InferenceService

BASE_REVISION = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
TOKENIZER_REVISION = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

ALL_TASKS = [
    "transaction_review",
    "variance_analysis",
    "cash_reconciliation",
]

ARM_SPECS: tuple[dict[str, Any], ...] = (
    {
        "arm_id": "student_base",
        "model_id": "model_student_base",
        "artifact_id": "artifact_student_base",
        "kind": "base",
        "relative_path": "base",
        "display_name": "Base student",
        "purpose": "Frozen student starting point",
        "recipe": None,
        "promotion_status": "unknown",
    },
    {
        "arm_id": "oracle_sft",
        "model_id": "model_oracle_sft",
        "artifact_id": "artifact_oracle_sft",
        "kind": "peft_adapter",
        "relative_path": "adapters/oracle_sft",
        "display_name": "Oracle SFT",
        "purpose": "Oracle-gold sequence upper bound",
        "recipe": "oracle_sft.v1",
        "promotion_status": "unknown",
    },
    {
        "arm_id": "sequence_kd",
        "model_id": "model_sequence_kd",
        "artifact_id": "artifact_sequence_kd",
        "kind": "peft_adapter",
        "relative_path": "adapters/sequence_kd",
        "display_name": "Sequence KD",
        "purpose": "sequence.v1 treatment",
        "recipe": "sequence.v1",
        "promotion_status": "unknown",
    },
    {
        "arm_id": "logit_kd",
        "model_id": "model_logit_kd",
        "artifact_id": "artifact_logit_kd",
        "kind": "peft_adapter",
        "relative_path": "adapters/logit_kd",
        "display_name": "Logit KD",
        "purpose": "logit.v1 treatment",
        "recipe": "logit.v1",
        "promotion_status": "unknown",
    },
    {
        "arm_id": "ce_ablation",
        "model_id": "model_ce_ablation",
        "artifact_id": "artifact_ce_ablation",
        "kind": "peft_adapter",
        "relative_path": "adapters/ce_ablation",
        "display_name": "CE ablation",
        "purpose": "Matched CE-only control",
        "recipe": "ce_ablation.v1",
        "promotion_status": "unknown",
    },
    {
        "arm_id": "promoted_winner",
        "model_id": "model_promoted_winner",
        "artifact_id": "artifact_promoted_winner",
        "kind": "peft_adapter",
        "relative_path": "adapters/promoted_winner",
        "display_name": "Promoted winner",
        "purpose": "Proof-promoted serving candidate",
        "recipe": "sequence.v1",
        "promotion_status": "promoted",
        "proof_status": "proved",
    },
)


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def build_bundle(
    root: Path,
    *,
    exclude_arms: set[str] | None = None,
    corrupt_checksum_for: str | None = None,
    omit_model: str | None = None,
) -> Path:
    exclude_arms = exclude_arms or set()
    checksums: dict[str, str] = {}
    artifacts: list[dict[str, Any]] = []

    base_config = {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}
    checksums["base/config.json"] = _write(
        root / "base" / "config.json",
        json.dumps(base_config, sort_keys=True).encode("utf-8"),
    )
    checksums["base/tokenizer.json"] = _write(
        root / "base" / "tokenizer.json",
        b'{"model":"fake-tokenizer"}',
    )
    checksums["base/weights.marker"] = _write(
        root / "base" / "weights.marker",
        b"fake-base-weights",
    )

    for spec in ARM_SPECS:
        if omit_model == spec["model_id"]:
            continue
        relative = spec["relative_path"]
        if spec["kind"] == "base":
            file_rel = "config.json"
            digest = checksums["base/config.json"]
            artifact_checksums = {file_rel: digest}
        else:
            file_rel = "adapter_model.safetensors"
            digest = _write(
                root / relative / file_rel,
                f"fake-adapter:{spec['arm_id']}".encode(),
            )
            checksums[f"{relative}/{file_rel}"] = digest
            config_digest = _write(
                root / relative / "adapter_config.json",
                json.dumps({"peft_type": "LORA", "arm": spec["arm_id"]}).encode("utf-8"),
            )
            checksums[f"{relative}/adapter_config.json"] = config_digest
            artifact_checksums = {
                file_rel: digest,
                "adapter_config.json": config_digest,
            }
        excluded = spec["arm_id"] in exclude_arms
        artifacts.append(
            {
                "schema_version": "distillery.serving_artifact.v1",
                "artifact_id": spec["artifact_id"],
                "model_id": spec["model_id"],
                "arm_id": spec["arm_id"],
                "kind": spec["kind"],
                "relative_path": relative,
                "display_name": spec["display_name"],
                "purpose": spec["purpose"],
                "base_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "base_revision": BASE_REVISION,
                "tokenizer_revision": TOKENIZER_REVISION,
                "supported_tasks": list(ALL_TASKS),
                "checksums": artifact_checksums,
                "recipe": spec.get("recipe"),
                "proof_status": spec.get("proof_status"),
                "promotion_status": spec.get("promotion_status", "unknown"),
                "excluded": excluded,
                "exclusion_reason": "excluded for test" if excluded else None,
                "stats": {
                    "advertised_parameter_count": 494_000_000,
                    "adapter_parameter_count": 0 if spec["kind"] == "base" else 8_400_000,
                    "compression_ratio": 3.117,
                    "seed": 17,
                    "artifact_hash": digest,
                    "teacher": {
                        "id": "Qwen/Qwen2.5-1.5B-Instruct",
                        "revision": TOKENIZER_REVISION,
                    },
                    "student": {
                        "id": "Qwen/Qwen2.5-0.5B-Instruct",
                        "revision": BASE_REVISION,
                    },
                },
            }
        )

    registry = {
        "schema_version": "distillery.serving_registry.v1",
        "run_id": "run_inference_fixture_001",
        "dataset_id": "dataset_finance_world_v1",
        "endpoint_id": "distillery-demo-inference",
        "base_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "base_revision": BASE_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "base_relative_path": "base",
        "artifacts": artifacts,
    }
    registry_bytes = (json.dumps(registry, indent=2, sort_keys=True) + "\n").encode("utf-8")
    checksums["serving_registry.json"] = _write(root / "serving_registry.json", registry_bytes)

    if corrupt_checksum_for is not None:
        checksums[corrupt_checksum_for] = "0" * 64

    sums_lines = [f"{digest}  {rel}" for rel, digest in sorted(checksums.items())]
    (root / "integrity").mkdir(parents=True, exist_ok=True)
    (root / "integrity" / "SHA256SUMS").write_text(
        "\n".join(sums_lines) + "\n",
        encoding="utf-8",
    )
    return root


def make_settings(bundle_root: Path, **overrides: Any) -> InferenceSettings:
    values: dict[str, Any] = {
        "model_bundle_root": bundle_root,
        "endpoint_id": "test-endpoint",
        "max_prompt_tokens": 256,
        "max_completion_tokens": 128,
        "max_input_bytes": 8192,
        "request_timeout_s": 2.0,
        "max_concurrent_requests": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 17,
        "require_offline": True,
        "runtime_backend": "fake",
    }
    values.update(overrides)
    return InferenceSettings(**values)


def make_service(
    tmp_path: Path,
    *,
    runtime: FakeRuntime | None = None,
    exclude_arms: set[str] | None = None,
    settings_overrides: dict[str, Any] | None = None,
) -> tuple[InferenceService, LoadedBundle, FakeRuntime]:
    bundle_root = build_bundle(tmp_path / "bundle", exclude_arms=exclude_arms)
    bundle = load_serving_bundle(bundle_root)
    fake = runtime or FakeRuntime(bundle=bundle)
    settings = make_settings(bundle_root, **(settings_overrides or {}))
    service = InferenceService(settings=settings, bundle=bundle, runtime=fake)
    return service, bundle, fake


def sample_input(task: str = "transaction_review") -> dict[str, Any]:
    if task == "transaction_review":
        return {
            "amount_minor": 189900,
            "currency": "USD",
            "descriptor": "ACME CLOUD ANNUAL",
            "gl_candidates": ["6400", "6100", "2100"],
            "vendor": "Acme Cloud",
        }
    if task == "variance_analysis":
        return {"drivers": [], "period": "2026-Q2"}
    return {
        "book_balance_minor": 1000,
        "bank_balance_minor": 1000,
        "transactions": [],
    }
