"""Schema/manifest round trips for technique descriptors."""

from __future__ import annotations

import json
from pathlib import Path

from distillery.contracts.hashing import content_sha256
from distillery.techniques import TechniqueDescriptor
from distillery.techniques.builtins import sequence_v1_descriptor


def test_seal_roundtrip_preserves_hash() -> None:
    original = sequence_v1_descriptor()
    payload = original.model_dump(mode="json")
    restored = TechniqueDescriptor.model_validate(payload)
    assert restored.descriptor_sha256 == original.descriptor_sha256
    assert restored.canonical_payload() == original.canonical_payload()
    assert content_sha256(restored.canonical_payload()) == restored.descriptor_sha256


def test_json_file_roundtrip(tmp_path: Path) -> None:
    original = sequence_v1_descriptor()
    path = tmp_path / "sequence.json"
    path.write_text(json.dumps(original.model_dump(mode="json")), encoding="utf-8")
    restored = TechniqueDescriptor.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert restored.technique_key == "sequence.v1@1.0.0"
    restored.assert_integrity()


def test_external_example_builder_roundtrip() -> None:
    from examples.byodt.reverse_kl_v1.build_descriptor import build

    descriptor = build()
    again = TechniqueDescriptor.model_validate(descriptor.model_dump(mode="json"))
    assert again.descriptor_sha256 == descriptor.descriptor_sha256
    assert again.execution.value == "external_container"
    assert again.plugin_image is not None
    assert "@sha256:" in again.plugin_image.image_uri
