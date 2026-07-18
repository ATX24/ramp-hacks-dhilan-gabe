"""Content-addressed external channel identity and tamper regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distillery.contracts.hashing import content_sha256
from distillery.techniques import (
    TechniqueError,
    TechniqueErrorCode,
    TechniqueRegistry,
    TechniqueRequest,
)
from distillery.techniques.channel import load_channel_plan, write_channel_plan


def _external_plan(
    registry: TechniqueRegistry,
    external_descriptor,
    external_config: dict,
    logit_context,
):
    registry.register(external_descriptor)
    return registry.plan(
        TechniqueRequest(
            technique_id=external_descriptor.technique_id,
            version=external_descriptor.version,
            config=external_config,
        ),
        logit_context,
    )


def _rewrite_envelope(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    canonical = {key: value for key, value in payload.items() if key != "envelope_sha256"}
    payload["envelope_sha256"] = content_sha256(canonical)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_channel_contains_config_and_complete_identity(
    tmp_path: Path,
    registry,
    external_descriptor,
    external_config,
    logit_context,
) -> None:
    plan = _external_plan(registry, external_descriptor, external_config, logit_context)
    channel = tmp_path / "channel"
    write_channel_plan(channel, plan=plan)
    envelope = load_channel_plan(channel)
    assert dict(envelope.config) == external_config
    assert content_sha256(dict(envelope.config)) == envelope.contract.config_sha256
    assert envelope.contract.protocol_sha256 == plan.protocol_sha256
    assert envelope.contract.environment_sha256 == content_sha256(
        logit_context.model_dump(mode="json")
    )
    assert envelope.contract.image_uri == plan.external_execution.image_uri
    assert envelope.contract.reviewed_source_commit == (
        plan.external_execution.reviewed_source_commit
    )
    assert envelope.envelope_sha256 == content_sha256(envelope.canonical_payload())


@pytest.mark.parametrize("sidecar", ["README", "sidecar.bin", "other.json"])
def test_channel_rejects_every_sidecar(
    sidecar: str,
    tmp_path: Path,
    registry,
    external_descriptor,
    external_config,
    logit_context,
) -> None:
    plan = _external_plan(registry, external_descriptor, external_config, logit_context)
    channel = tmp_path / "channel"
    write_channel_plan(channel, plan=plan)
    (channel / sidecar).write_bytes(b"x")
    with pytest.raises(TechniqueError) as excinfo:
        load_channel_plan(channel)
    assert excinfo.value.code is TechniqueErrorCode.TECHNIQUE_CHANNEL_INVALID


def test_channel_rejects_directory_sidecar(
    tmp_path: Path,
    registry,
    external_descriptor,
    external_config,
    logit_context,
) -> None:
    plan = _external_plan(registry, external_descriptor, external_config, logit_context)
    channel = tmp_path / "channel"
    write_channel_plan(channel, plan=plan)
    (channel / "sidecar").mkdir()
    with pytest.raises(TechniqueError):
        load_channel_plan(channel)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["contract"].__setitem__("protocol_sha256", "0" * 64),
        lambda p: p["contract"].__setitem__(
            "image_uri",
            (f"123456789012.dkr.ecr.us-east-1.amazonaws.com/other@sha256:{'c' * 64}"),
        ),
        lambda p: p["contract"].__setitem__("reviewed_source_commit", "0" * 40),
        lambda p: p["contract"].__setitem__("artifact_contract_sha256", "0" * 64),
        lambda p: p["config"].__setitem__("temperature", 9.0),
    ],
)
def test_channel_rejects_contract_config_and_execution_tamper(
    mutate,
    tmp_path: Path,
    registry,
    external_descriptor,
    external_config,
    logit_context,
) -> None:
    plan = _external_plan(registry, external_descriptor, external_config, logit_context)
    path = write_channel_plan(tmp_path / "channel", plan=plan)
    _rewrite_envelope(path, mutate)
    with pytest.raises(TechniqueError):
        load_channel_plan(path.parent)


def test_channel_rejects_plan_protocol_mismatch(
    tmp_path: Path,
    registry,
    external_descriptor,
    external_config,
    logit_context,
) -> None:
    plan = _external_plan(registry, external_descriptor, external_config, logit_context)
    path = write_channel_plan(tmp_path / "channel", plan=plan)
    _rewrite_envelope(
        path,
        lambda payload: payload["plan"].__setitem__("protocol_sha256", "0" * 64),
    )
    with pytest.raises(TechniqueError) as excinfo:
        load_channel_plan(path.parent)
    assert excinfo.value.code is TechniqueErrorCode.TECHNIQUE_NONDETERMINISTIC


def test_channel_rejects_unexpected_fields_and_overwrite(
    tmp_path: Path,
    registry,
    external_descriptor,
    external_config,
    logit_context,
) -> None:
    plan = _external_plan(registry, external_descriptor, external_config, logit_context)
    channel = tmp_path / "channel"
    path = write_channel_plan(channel, plan=plan)
    with pytest.raises(TechniqueError):
        write_channel_plan(channel, plan=plan)
    _rewrite_envelope(path, lambda payload: payload.__setitem__("extras", {}))
    with pytest.raises(TechniqueError):
        load_channel_plan(channel)
