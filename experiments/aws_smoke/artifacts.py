"""Artifact layout helpers and checksum verification for emergency runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from distillery.training.artifacts import (
    RUN_ARTIFACT_LAYOUT,
    build_run_artifact_layout,
    parse_sha256sums,
    sha256_file,
    verify_sha256sums,
    write_sha256sums,
)

EMERGENCY_REQUIRED_RELATIVE = (
    RUN_ARTIFACT_LAYOUT["manifest"],
    RUN_ARTIFACT_LAYOUT["training_metrics"],
    "training/final/adapter_config.json",
    RUN_ARTIFACT_LAYOUT["integrity_sums"],
    "training/emergency_run.json",
)


def expected_layout(*, run_id: str, root_prefix: str) -> dict[str, Any]:
    layout = build_run_artifact_layout(run_id=run_id, root_prefix=root_prefix)
    return {
        "run_id": run_id,
        "root_prefix": root_prefix,
        "files": [f.model_dump(mode="json") for f in layout.files],
        "emergency_required": list(EMERGENCY_REQUIRED_RELATIVE),
    }


def write_emergency_integrity(
    root: Path,
    *,
    extra_entries: dict[str, str] | None = None,
) -> Path:
    entries: dict[str, str] = {}
    for rel in EMERGENCY_REQUIRED_RELATIVE:
        if rel == RUN_ARTIFACT_LAYOUT["integrity_sums"]:
            continue
        path = root / rel
        if path.is_file():
            entries[rel] = sha256_file(path)
    if extra_entries:
        entries.update(extra_entries)
    destination = root / RUN_ARTIFACT_LAYOUT["integrity_sums"]
    return write_sha256sums(entries, destination=destination)


def verify_emergency_artifacts(root: Path) -> dict[str, Any]:
    """Verify SHA256SUMS and required emergency files. Pure local filesystem."""
    sums_path = root / RUN_ARTIFACT_LAYOUT["integrity_sums"]
    if not sums_path.is_file():
        raise FileNotFoundError(f"missing integrity file: {sums_path}")
    entries = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    missing_required = [
        rel for rel in EMERGENCY_REQUIRED_RELATIVE if rel != sums_path.name and rel not in entries
    ]
    # integrity file itself is listed in EMERGENCY_REQUIRED_RELATIVE by layout key path
    missing_required = [
        rel
        for rel in EMERGENCY_REQUIRED_RELATIVE
        if rel != RUN_ARTIFACT_LAYOUT["integrity_sums"] and not (root / rel).is_file()
    ]
    if missing_required:
        raise FileNotFoundError(
            "missing required emergency artifacts: " + ", ".join(missing_required)
        )
    verify_sha256sums(entries, root=root)
    report = {
        "ok": True,
        "root": str(root),
        "checked": sorted(entries),
        "count": len(entries),
    }
    report_path = root / "integrity/verify_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report
