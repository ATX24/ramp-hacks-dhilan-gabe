"""Offline File-mode channels for sealed 72B jobs (no network pulls)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillery.contracts.hashing import content_sha256

CANONICAL_MANIFEST_FILENAME = "manifest.json"
CANONICAL_ORACLE_FILENAME = "oracle_sequences.json"
CANONICAL_TRAJECTORIES_FILENAME = "precomputed_trajectories.json"
CANONICAL_DATASET_FILENAME = "train.jsonl"

CHANNEL_NAMES = (
    "manifest",
    "dataset",
    "models",
    "oracle_sequences",
    "precomputed_trajectories",
)


@dataclass(frozen=True, slots=True)
class OfflineChannels:
    root: Path
    manifest: Path
    dataset: Path
    models: Path
    oracle_sequences: Path
    precomputed_trajectories: Path

    def as_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "distillery.qwen72b_fallback.channels.v1",
            "mode": "offline_file",
            "channels": {
                "manifest": self.manifest.as_posix(),
                "dataset": self.dataset.as_posix(),
                "models": self.models.as_posix(),
                "oracle_sequences": self.oracle_sequences.as_posix(),
                "precomputed_trajectories": self.precomputed_trajectories.as_posix(),
            },
            "network": "disabled",
            "hf_hub_offline": True,
            "transformers_offline": True,
            "trajectories_on_warm_timer": False,
        }


def default_sm_channels(root: Path | None = None) -> OfflineChannels:
    base = root or Path("/opt/ml/input/data")
    return OfflineChannels(
        root=base,
        manifest=base / "manifest",
        dataset=base / "dataset",
        models=base / "models",
        oracle_sequences=base / "oracle_sequences",
        precomputed_trajectories=base / "precomputed_trajectories",
    )


def discover_single_json(channel: Path, *, filename: str) -> Path:
    if not channel.is_dir():
        raise FileNotFoundError(f"channel directory missing: {channel}")
    json_files = sorted(path for path in channel.iterdir() if path.suffix == ".json")
    canonical = channel / filename
    if json_files != [canonical] or not canonical.is_file():
        raise FileNotFoundError(
            f"channel {channel} must contain exactly one JSON file named {filename!r}; "
            f"found={[path.name for path in json_files]}"
        )
    return canonical


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def channel_contract_hash(contract: dict[str, Any]) -> str:
    return content_sha256(contract)
