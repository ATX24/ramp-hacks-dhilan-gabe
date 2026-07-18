"""Sealed artifact layout and SHA256SUMS helpers for 72B runs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillery.contracts.hashing import content_sha256

ARTIFACT_RELATIVE_PATHS = (
    "model/adapter/adapter_config.json",
    "model/adapter/adapter_model.safetensors",
    "integrity/SHA256SUMS",
    "manifest.json",
    "protocol.json",
    "memory_plan.json",
    "gross_cost.json",
)


@dataclass(frozen=True, slots=True)
class SealedArtifactLayout:
    root: Path

    def path_for(self, relative: str) -> Path:
        if relative.startswith("/") or ".." in relative.split("/"):
            raise ValueError(f"unsafe artifact relative path: {relative}")
        return self.root / relative


def write_sha256sums(entries: dict[str, str], *, destination: Path) -> Path:
    lines = [f"{digest}  {rel}" for rel, digest in sorted(entries.items())]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def seal_run_artifacts(
    root: Path,
    *,
    adapter_config: dict[str, Any],
    adapter_blob: bytes,
    protocol: dict[str, Any],
    memory_plan: dict[str, Any],
    gross_cost: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, str]:
    layout = SealedArtifactLayout(root)
    files: dict[str, bytes] = {
        "model/adapter/adapter_config.json": (
            __import__("json").dumps(adapter_config, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "model/adapter/adapter_model.safetensors": adapter_blob,
        "protocol.json": (
            __import__("json").dumps(protocol, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "memory_plan.json": (
            __import__("json").dumps(memory_plan, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "gross_cost.json": (
            __import__("json").dumps(gross_cost, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "manifest.json": (
            __import__("json").dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode(),
    }
    checksums = {rel: sha256_bytes(blob) for rel, blob in files.items()}
    for rel, blob in files.items():
        path = layout.path_for(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    write_sha256sums(checksums, destination=layout.path_for("integrity/SHA256SUMS"))
    checksums["integrity/SHA256SUMS"] = sha256_bytes(
        layout.path_for("integrity/SHA256SUMS").read_bytes()
    )
    return checksums


def artifact_bundle_sha256(checksums: dict[str, str]) -> str:
    return content_sha256({"checksums": checksums})
