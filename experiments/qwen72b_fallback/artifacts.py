"""Seal only a real saved/reloaded PEFT adapter, never caller-provided bytes."""

from __future__ import annotations

import json
from pathlib import Path

from distillery.contracts.hashing import content_sha256
from experiments.qwen72b_fallback.evidence import sha256_file

REQUIRED_SUCCESS_PATHS = (
    "model/adapter/adapter_config.json",
    "model/adapter/adapter_model.safetensors",
    "reload-probe.json",
    "protocol.json",
    "profile.json",
    "run-evidence.json",
    "compliance/QWEN_NOTICE.txt",
    "compliance/attribution_plan.json",
)


def write_sha256sums(entries: dict[str, str], *, destination: Path) -> Path:
    lines = [f"{digest}  {relative}" for relative, digest in sorted(entries.items())]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def artifact_bundle_sha256(checksums: dict[str, str]) -> str:
    return content_sha256({"checksums": checksums})


def validate_real_adapter(adapter_dir: Path) -> dict[str, str]:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError("real PEFT adapter files are missing")
    config = json.loads(config_path.read_bytes())
    if not isinstance(config, dict) or not config.get("peft_type"):
        raise RuntimeError("adapter_config.json is not a PEFT adapter config")
    try:
        from safetensors import safe_open

        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            names = list(handle.keys())
            if not names:
                raise RuntimeError("adapter safetensors contains no tensors")
            if any(handle.get_tensor(name).numel() == 0 for name in names):
                raise RuntimeError("adapter safetensors contains an empty tensor")
    except Exception as exc:
        raise RuntimeError(f"adapter safetensors validation failed: {exc}") from exc
    return {
        config_path.name: sha256_file(config_path),
        weights_path.name: sha256_file(weights_path),
    }


def seal_existing_run_artifacts(root: Path) -> dict[str, str]:
    """Validate real PEFT/reload evidence, then hash every existing run artifact."""
    missing = [relative for relative in REQUIRED_SUCCESS_PATHS if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"success artifact set is incomplete: {missing}")
    validate_real_adapter(root / "model" / "adapter")
    reload_report = json.loads((root / "reload-probe.json").read_bytes())
    required_reload = {
        "fresh_base_loaded": True,
        "peft_adapter_reloaded": True,
        "forward_finite": True,
    }
    if any(reload_report.get(key) != value for key, value in required_reload.items()):
        raise RuntimeError("real PEFT reload/forward probe did not pass")
    checksums = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix()
        not in {"integrity/SHA256SUMS", "bundle.json", "completion/all-ranks.json"}
    }
    write_sha256sums(
        checksums,
        destination=root / "integrity" / "SHA256SUMS",
    )
    checksums["integrity/SHA256SUMS"] = sha256_file(root / "integrity" / "SHA256SUMS")
    return checksums
