"""Canonical SageMaker File-mode manifest channel contract."""

from __future__ import annotations

import json
from pathlib import Path

from distillery.contracts.manifest import SealedRunManifest

CANONICAL_MANIFEST_FILENAME = "manifest.json"


def load_manifest(path: Path) -> SealedRunManifest:
    if path.name != CANONICAL_MANIFEST_FILENAME:
        raise ValueError(
            f"manifest must use canonical filename {CANONICAL_MANIFEST_FILENAME!r}"
        )
    return SealedRunManifest.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )


def discover_manifest(channel: Path) -> Path:
    """Require exactly one canonical JSON file, rejecting ambiguity."""
    if not channel.is_dir():
        raise FileNotFoundError(f"manifest channel directory missing: {channel}")
    json_files = sorted(path for path in channel.iterdir() if path.suffix == ".json")
    canonical = channel / CANONICAL_MANIFEST_FILENAME
    if json_files != [canonical] or not canonical.is_file():
        raise FileNotFoundError(
            "manifest channel must contain exactly one JSON file named "
            f"{CANONICAL_MANIFEST_FILENAME}; found={[path.name for path in json_files]}"
        )
    return canonical


def discover_and_load_manifest(channel: Path) -> SealedRunManifest:
    return load_manifest(discover_manifest(channel))
