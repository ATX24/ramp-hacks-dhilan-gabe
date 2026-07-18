"""Strict emergency artifact contract and SHA-256 verification."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from distillery.contracts.manifest import SealedRunManifest
from distillery.training.artifacts import (
    RUN_ARTIFACT_LAYOUT,
    build_run_artifact_layout,
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
TOKENIZER_DATA_CANDIDATES = (
    ADAPTER_DIR / "tokenizer.json",
    ADAPTER_DIR / "tokenizer.model",
    ADAPTER_DIR / "sentencepiece.bpe.model",
)
REQUIRED_RELATIVE = (
    Path(RUN_ARTIFACT_LAYOUT["manifest"]),
    Path(RUN_ARTIFACT_LAYOUT["training_metrics"]),
    Path("evaluation/predictions.jsonl"),
    ADAPTER_DIR / "adapter_config.json",
    ADAPTER_DIR / "tokenizer_config.json",
    Path("model/tokenizer_evidence.json"),
    Path("model/chat_template.txt"),
    Path("model/load_test.json"),
    Path("costs/gross_cost.json"),
    Path("training/emergency_run.json"),
)
INTEGRITY_RELATIVE = Path(RUN_ARTIFACT_LAYOUT["integrity_sums"])


def expected_layout(*, run_id: str, root_prefix: str) -> dict[str, Any]:
    layout = build_run_artifact_layout(run_id=run_id, root_prefix=root_prefix)
    return {
        "run_id": run_id,
        "root_prefix": root_prefix,
        "files": [item.model_dump(mode="json") for item in layout.files],
        "emergency_required": [path.as_posix() for path in REQUIRED_RELATIVE],
        "adapter_weight_candidates": [
            path.as_posix() for path in ADAPTER_WEIGHT_CANDIDATES
        ],
    }


def write_emergency_integrity(root: Path) -> Path:
    """Refuse incomplete output, then checksum every produced regular file."""
    _validate_artifact_payloads(root)
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


def verify_emergency_artifacts(root: Path) -> dict[str, Any]:
    """Require complete, nonempty, reload-tested and fully checksummed output."""
    sums_path = root / INTEGRITY_RELATIVE
    if not sums_path.is_file() or sums_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing nonempty integrity file: {sums_path}")
    _validate_artifact_payloads(root)
    entries = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    verify_sha256sums(entries, root=root)

    produced = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != sums_path
    }
    missing_checksums = sorted(produced - set(entries))
    stale_checksums = sorted(set(entries) - produced)
    if missing_checksums or stale_checksums:
        raise ValueError(
            "SHA256SUMS must exactly cover produced artifacts; "
            f"missing={missing_checksums} stale={stale_checksums}"
        )
    weight_path = _first_nonempty(root, ADAPTER_WEIGHT_CANDIDATES)
    return {
        "ok": True,
        "root": str(root),
        "checked": sorted(entries),
        "count": len(entries),
        "adapter_weights": str(weight_path.relative_to(root)),
        "preferred_safetensors": weight_path.name == "adapter_model.safetensors",
    }


def _validate_artifact_payloads(root: Path) -> None:
    missing = [
        path.as_posix()
        for path in REQUIRED_RELATIVE
        if not _is_nonempty_file(root / path)
    ]
    if missing:
        raise FileNotFoundError(
            "missing required nonempty emergency artifacts: " + ", ".join(missing)
        )
    _first_nonempty(root, ADAPTER_WEIGHT_CANDIDATES)
    if not any(_is_nonempty_file(root / path) for path in TOKENIZER_DATA_CANDIDATES):
        # Qwen fast tokenizer must emit tokenizer.json. The alternatives support
        # an explicitly sealed tokenizer-model representation.
        raise FileNotFoundError("missing nonempty tokenizer data artifact")

    manifest = _read_json(root / RUN_ARTIFACT_LAYOUT["manifest"])
    SealedRunManifest.model_validate(manifest)

    adapter_config = _read_json(root / ADAPTER_DIR / "adapter_config.json")
    if not adapter_config.get("peft_type") or not adapter_config.get("target_modules"):
        raise ValueError("adapter_config.json lacks PEFT type or target modules")

    tokenizer_evidence = _read_json(root / "model/tokenizer_evidence.json")
    if tokenizer_evidence.get("compatible") is not True:
        raise ValueError("tokenizer evidence does not attest loaded compatibility")
    template = (root / "model/chat_template.txt").read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError("chat template artifact is empty")

    metrics = _read_jsonl(root / RUN_ARTIFACT_LAYOUT["training_metrics"])
    if not metrics or any(not math.isfinite(float(row["loss"])) for row in metrics):
        raise ValueError("metrics must contain finite loss records")
    predictions = _read_jsonl(root / "evaluation/predictions.jsonl")
    if not predictions:
        raise ValueError("predictions artifact is empty")
    for row in predictions:
        if not row.get("example_id") or "prediction_text" not in row:
            raise ValueError("prediction row lacks example_id or prediction_text")

    cost = _read_json(root / "costs/gross_cost.json")
    gross = float(cost.get("gross_cost_usd", -1))
    ceiling = float(cost.get("max_run_usd", -1))
    if not math.isfinite(gross) or gross < 0 or gross > ceiling or ceiling <= 0:
        raise ValueError("gross cost evidence is malformed or exceeds the run ceiling")

    load_test = _read_json(root / "model/load_test.json")
    if (
        load_test.get("passed") is not True
        or load_test.get("fresh_base_loaded") is not True
        or load_test.get("adapter_reloaded") is not True
        or load_test.get("forward_finite") is not True
    ):
        raise ValueError("adapter load-test evidence did not prove fresh reload + forward")

    run = _read_json(root / "training/emergency_run.json")
    if run.get("status") != "completed" or int(run.get("completed_steps", 0)) < 1:
        raise ValueError("emergency run report does not attest completed training")


def _first_nonempty(root: Path, candidates: tuple[Path, ...]) -> Path:
    for relative in candidates:
        path = root / relative
        if _is_nonempty_file(path):
            return path
    raise FileNotFoundError(
        "missing nonempty adapter weights; expected adapter_model.safetensors "
        "or adapter_model.bin"
    )


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and path.stat().st_size > 0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object on {path}:{line_number}")
        rows.append(payload)
    return rows
