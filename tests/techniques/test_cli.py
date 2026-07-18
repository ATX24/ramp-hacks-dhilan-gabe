"""CLI seam: validate / register / plan-only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

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


def test_cli_validate_and_plan(tmp_path: Path, example_technique_json: Path) -> None:
    validate = _run("validate", str(example_technique_json))
    assert validate.returncode == 0, validate.stderr
    payload = json.loads(validate.stdout)
    assert payload["ok"] is True
    assert payload["technique_id"] == "hackathon.dhilan.reverse_kl"

    registry_dir = tmp_path / "registry"
    register = _run(
        "register",
        str(example_technique_json),
        "--registry-dir",
        str(registry_dir),
    )
    assert register.returncode == 0, register.stderr

    config = tmp_path / "config.json"
    context = tmp_path / "context.json"
    config.write_text(
        json.dumps(
            {
                "max_length": 512,
                "max_completion": 160,
                "seed": 17,
                "temperature": 2.0,
            }
        ),
        encoding="utf-8",
    )
    context.write_text(
        json.dumps(
            {
                "backend_kind": "sagemaker",
                "student_model_id": "Qwen/Qwen2.5-0.5B",
                "student_revision": "a" * 40,
                "teacher_model_id": "Qwen/Qwen2.5-1.5B",
                "teacher_revision": "b" * 40,
                "tokenizer_sha256_student": "1" * 64,
                "tokenizer_sha256_teacher": "1" * 64,
                "chat_template_sha256_student": "2" * 64,
                "chat_template_sha256_teacher": "2" * 64,
                "special_token_map_match": True,
                "local_white_box": True,
                "network_isolation": True,
                "instance_type": "ml.g5.xlarge",
            }
        ),
        encoding="utf-8",
    )
    channel_dir = tmp_path / "channel"
    plan = _run(
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
        str(channel_dir),
    )
    assert plan.returncode == 0, plan.stderr
    body = json.loads(plan.stdout)
    assert body["lifecycle"] == "planned"
    assert body["external_execution"]["import_forbidden"] is True
    assert (channel_dir / "technique_plan.json").is_file()


def test_cli_validate_rejects_bad_hash(tmp_path: Path, example_technique_json: Path) -> None:
    payload = json.loads(example_technique_json.read_text(encoding="utf-8"))
    payload["summary"] = "tampered"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    result = _run("validate", str(bad))
    assert result.returncode == 2
    err = json.loads(result.stderr)
    assert err["ok"] is False
    assert err["code"] == "TECHNIQUE_DESCRIPTOR_INVALID"
