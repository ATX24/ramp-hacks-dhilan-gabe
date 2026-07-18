"""Sealed PEFT adapter artifacts, integrity manifest, and reload smoke."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from distillery.contracts.hashing import content_sha256, sha256_hex
from distillery.training.artifacts import (
    parse_sha256sums,
    sha256_file,
    verify_sha256sums,
    write_sha256sums,
)

ADAPTER_DIR = Path("model/adapter")
ADAPTER_WEIGHT_CANDIDATES = (
    ADAPTER_DIR / "adapter_model.safetensors",
    ADAPTER_DIR / "adapter_model.bin",
)
REQUIRED_RELATIVE = (
    Path("manifest/manifest.json"),
    Path("training/metrics.jsonl"),
    Path("training/huge_backup_run.json"),
    Path("evaluation/smoke_predictions.jsonl"),
    ADAPTER_DIR / "adapter_config.json",
    Path("model/load_test.json"),
    Path("costs/gross_cost.json"),
    Path("protocol/protocol.json"),
)
INTEGRITY_RELATIVE = Path("SHA256SUMS")


class ArtifactError(ValueError):
    pass


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_adapter_stub(
    root: Path,
    *,
    base_model_name_or_path: str,
    lora_rank: int,
    target_modules: list[str],
    weight_bytes: bytes | None = None,
) -> Path:
    adapter_dir = root / ADAPTER_DIR
    adapter_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model_name_or_path,
        "r": lora_rank,
        "lora_alpha": lora_rank * 2,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "bias": "none",
    }
    write_json(adapter_dir / "adapter_config.json", config)
    weight_path = adapter_dir / "adapter_model.bin"
    payload = weight_bytes or b"HUGE_BACKUP_FAKE_ADAPTER_WEIGHTS_v1"
    weight_path.write_bytes(payload)
    return weight_path


def write_integrity_manifest(root: Path) -> Path:
    _validate_required(root)
    entries: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == INTEGRITY_RELATIVE.as_posix():
            continue
        entries[relative] = sha256_file(path)
    destination = root / INTEGRITY_RELATIVE
    return write_sha256sums(entries, destination=destination)


def verify_huge_backup_artifacts(root: Path) -> dict[str, Any]:
    sums_path = root / INTEGRITY_RELATIVE
    if not sums_path.is_file() or sums_path.stat().st_size == 0:
        raise ArtifactError(f"missing nonempty integrity file: {sums_path}")
    _validate_required(root)
    entries = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    verify_sha256sums(entries, root=root)
    produced = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != sums_path
    }
    missing = sorted(produced - set(entries))
    stale = sorted(set(entries) - produced)
    if missing or stale:
        raise ArtifactError(f"SHA256SUMS coverage mismatch missing={missing} stale={stale}")
    weight = _first_nonempty(root, ADAPTER_WEIGHT_CANDIDATES)
    return {
        "ok": True,
        "root": str(root),
        "count": len(entries),
        "adapter_weights": str(weight.relative_to(root)),
        "integrity_sha256": sha256_hex(sums_path.read_bytes()),
    }


def _validate_required(root: Path) -> None:
    missing = [path.as_posix() for path in REQUIRED_RELATIVE if not _is_nonempty_file(root / path)]
    if missing:
        raise ArtifactError("missing required nonempty artifacts: " + ", ".join(missing))
    _first_nonempty(root, ADAPTER_WEIGHT_CANDIDATES)

    adapter_config = _read_json(root / ADAPTER_DIR / "adapter_config.json")
    if not adapter_config.get("peft_type") or not adapter_config.get("target_modules"):
        raise ArtifactError("adapter_config.json lacks PEFT type or target modules")

    cost = _read_json(root / "costs/gross_cost.json")
    gross = float(cost.get("gross_cost_usd", -1))
    ceiling = float(cost.get("max_run_usd", -1))
    if not math.isfinite(gross) or gross < 0 or gross > ceiling or ceiling <= 0:
        raise ArtifactError("gross cost evidence malformed or exceeds ceiling")

    load_test = _read_json(root / "model/load_test.json")
    if (
        load_test.get("passed") is not True
        or load_test.get("adapter_reloaded") is not True
        or load_test.get("fresh_base_loaded") is not True
    ):
        raise ArtifactError("adapter load-test evidence failed")

    run = _read_json(root / "training/huge_backup_run.json")
    if run.get("status") != "completed":
        raise ArtifactError("huge_backup run report not completed")

    protocol = _read_json(root / "protocol/protocol.json")
    if "protocol_hash" not in protocol:
        raise ArtifactError("protocol artifact missing protocol_hash")
    # Re-hash sealed objective block for tamper evidence.
    if protocol.get("objective_sha256") != content_sha256(protocol.get("objective", {})):
        raise ArtifactError("protocol objective_sha256 mismatch")


def _first_nonempty(root: Path, candidates: tuple[Path, ...]) -> Path:
    for relative in candidates:
        path = root / relative
        if _is_nonempty_file(path):
            return path
    raise ArtifactError("missing nonempty adapter weights")


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and path.stat().st_size > 0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload
