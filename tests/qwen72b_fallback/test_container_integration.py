from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from experiments.qwen72b_fallback.profile import rehearsal_profile
from experiments.qwen72b_fallback.readiness import ExecutionAction

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT / "containers" / "training" / "container_entrypoint.py"


def load_entrypoint():
    spec = importlib.util.spec_from_file_location("qwen72b_container_entrypoint", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hardened_wrapper_dispatches_only_to_isolated_gpu_launcher(
    authorization_factory,
    tmp_path: Path,
) -> None:
    profile = rehearsal_profile()
    launch_name = "qwen72b-rehearsal-wrapper"
    authorization = authorization_factory(
        action=ExecutionAction.REHEARSAL,
        profile=profile,
        launch_name=launch_name,
    )
    profile_path = tmp_path / "profile.json"
    auth_path = tmp_path / "authorization.json"
    probe_path = tmp_path / "memory-probe.json"
    profile_path.write_text(
        json.dumps(profile.model_dump(mode="json")),
        encoding="utf-8",
    )
    auth_path.write_text(
        json.dumps(authorization.model_dump(mode="json")),
        encoding="utf-8",
    )
    assert authorization.evidence_bundle.memory_probe is not None
    probe_path.write_text(
        json.dumps(authorization.evidence_bundle.memory_probe.model_dump(mode="json")),
        encoding="utf-8",
    )
    entrypoint = load_entrypoint()
    args = entrypoint.validate_trainer_arguments(
        [
            "--execution-mode",
            "qwen72b",
            "--qwen72b-mode",
            "rehearsal",
            "--launch-name",
            launch_name,
            "--profile",
            str(profile_path),
            "--authorization",
            str(auth_path),
            "--memory-probe",
            str(probe_path),
            "--models-dir",
            "/opt/ml/input/data/models",
            "--data-dir",
            "/opt/ml/input/data/data",
            "--output-dir",
            "/opt/ml/model",
            "--runtime-image-digest",
            authorization.evidence_bundle.ecr_image.image_digest,
            "--execute",
        ]
    )
    command = entrypoint.build_trainer_command(args)
    assert command[1:3] == [
        "-m",
        "experiments.qwen72b_fallback.distributed_launcher",
    ]
    assert "--runtime-image-digest" in command
    assert "--execute-acknowledgement" not in command
