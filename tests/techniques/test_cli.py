"""CLI descriptor validation, registration, and complete plan-only flow."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from distillery.techniques import TechniqueDescriptor

ROOT = Path(__file__).resolve().parents[2]
CTL = ROOT / "scripts" / "techniques" / "byodt_ctl.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(CTL), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cli_validate_register_and_plan(
    tmp_path: Path,
    example_technique_json: Path,
    external_config: dict,
    logit_context,
) -> None:
    validate = _run("validate-descriptor", str(example_technique_json))
    assert validate.returncode == 0, validate.stderr
    assert json.loads(validate.stdout)["technique_id"] == ("hackathon.dhilan.reverse_kl")
    registry_dir = tmp_path / "registry"
    register = _run(
        "register",
        str(example_technique_json),
        "--registry-dir",
        str(registry_dir),
    )
    assert register.returncode == 0, register.stderr
    config = _write_json(tmp_path / "config.json", external_config)
    context = _write_json(
        tmp_path / "context.json",
        logit_context.model_dump(mode="json"),
    )
    channel = tmp_path / "channel"
    result = _run(
        "plan",
        "--technique-id",
        "hackathon.dhilan.reverse_kl",
        "--version",
        "1.0.0",
        "--descriptor",
        str(example_technique_json),
        "--config",
        str(config),
        "--context",
        str(context),
        "--channel-dir",
        str(channel),
    )
    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert body["lifecycle"] == "planned"
    assert body["resolved_config"] == external_config
    assert body["external_execution"]["network_isolation_required"] is True
    assert body["channel_hash"]
    assert (channel / "technique_plan.json").is_file()


def test_cli_rejects_divergent_collision_independent_of_glob_order(
    tmp_path: Path,
    external_descriptor,
    external_config: dict,
    logit_context,
) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    divergent_payload = external_descriptor.canonical_payload()
    divergent_payload["summary"] = "different implementation"
    divergent = TechniqueDescriptor.seal(**divergent_payload)
    _write_json(
        registry_dir / "a.json",
        external_descriptor.model_dump(mode="json"),
    )
    _write_json(registry_dir / "z.json", divergent.model_dump(mode="json"))
    config = _write_json(tmp_path / "config.json", external_config)
    context = _write_json(
        tmp_path / "context.json",
        logit_context.model_dump(mode="json"),
    )
    result = _run(
        "plan",
        "--technique-id",
        external_descriptor.technique_id,
        "--version",
        external_descriptor.version,
        "--registry-dir",
        str(registry_dir),
        "--config",
        str(config),
        "--context",
        str(context),
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["code"] == "TECHNIQUE_VERSION_COLLISION"


def test_cli_validate_descriptor_rejects_bad_hash(
    tmp_path: Path,
    example_technique_json: Path,
) -> None:
    payload = json.loads(example_technique_json.read_text(encoding="utf-8"))
    payload["summary"] = "tampered"
    bad = _write_json(tmp_path / "bad.json", payload)
    result = _run("validate-descriptor", str(bad))
    assert result.returncode == 2
    assert json.loads(result.stderr)["code"] == "TECHNIQUE_DESCRIPTOR_INVALID"
