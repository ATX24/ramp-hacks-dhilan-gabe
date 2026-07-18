from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.aws_smoke.rescue_entry import (
    copy_verified_snapshot,
    verify_snapshot_tree,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def _write_snapshot(root: Path) -> dict[str, bytes]:
    files = {
        "config.json": b'{"model_type":"qwen2"}\n',
        "tokenizer_config.json": b'{"model_max_length":32768}\n',
        "tokenizer.json": b'{"version":"1.0"}\n',
        "weights/model-00001-of-00002.safetensors": b"first-weight-shard",
        "weights/model-00002-of-00002.safetensors": b"second-weight-shard",
    }
    evidence: dict[str, dict[str, object]] = {}
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        evidence[relative] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }
    (root / "snapshot-manifest.json").write_text(
        json.dumps(
            {
                "files": evidence,
                "model_id": MODEL_ID,
                "revision": REVISION,
            }
        ),
        encoding="utf-8",
    )
    return files


def test_snapshot_copy_preserves_paths_bytes_and_regular_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    expected = _write_snapshot(source)
    destination = tmp_path / "destination"

    copy_verified_snapshot(
        source,
        destination,
        expected_model_id=MODEL_ID,
        expected_revision=REVISION,
    )

    verify_snapshot_tree(
        destination,
        expected_model_id=MODEL_ID,
        expected_revision=REVISION,
    )
    for relative, content in expected.items():
        copied = destination / relative
        assert copied.is_file()
        assert not copied.is_symlink()
        assert copied.read_bytes() == content


def test_snapshot_preflight_rejects_any_symlink(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    _write_snapshot(snapshot)
    (snapshot / "forbidden-link").symlink_to(snapshot / "config.json")

    with pytest.raises(ValueError, match="forbidden symlink"):
        verify_snapshot_tree(
            snapshot,
            expected_model_id=MODEL_ID,
            expected_revision=REVISION,
        )
