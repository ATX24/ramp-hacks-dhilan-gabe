"""Static checks for inference container, scripts, and CloudFormation."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "containers" / "inference" / "Dockerfile"
ENTRYPOINT = ROOT / "containers" / "inference" / "container_entrypoint.py"
BUILD_SCRIPT = ROOT / "scripts" / "inference" / "build_inference_image.sh"
PUBLISH_SCRIPT = ROOT / "scripts" / "inference" / "publish_inference_image.sh"
DEPLOY_SCRIPT = ROOT / "infra" / "inference" / "deploy.sh"
TEMPLATE = ROOT / "infra" / "inference" / "template.yaml"
APP_DIR = ROOT / "apps" / "inference" / "distillery_inference"


def test_dockerfile_is_digest_pinned_non_root_offline() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert (
        "pytorch/pytorch@sha256:0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726"
        in text
    )
    assert 'distillery.runtime.uid="1000"' in text
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "useradd --uid 1000" in text
    assert "containers/training/ml-compatibility.json" in text
    assert "apps/inference" in text
    assert "EXPOSE 8080" in text
    assert "AKIA" not in text


def test_entrypoint_drops_privileges_and_imports_server() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "RUNTIME_UID = 1000" in text
    assert "drop_privileges" in text
    assert "distillery_inference.server" in text
    assert "uvicorn" in text
    tree = ast.parse(text)
    inline = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in tree.body
    ]
    assert inline == []


def test_inference_python_has_no_inline_imports() -> None:
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if node in tree.body:
                continue
            offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == []


def test_build_and_publish_default_dry_run() -> None:
    build = subprocess.run(
        ["bash", str(BUILD_SCRIPT), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert build.returncode == 0, build.stderr
    assert "dry_run=true" in build.stdout
    assert "docker_build=skipped" in build.stdout

    publish = subprocess.run(
        ["bash", str(PUBLISH_SCRIPT), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert publish.returncode == 0, publish.stderr
    assert "dry_run=true" in publish.stdout
    assert "docker_push=skipped" in publish.stdout


def test_deploy_script_defaults_to_plan() -> None:
    result = subprocess.run(
        ["bash", str(DEPLOY_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "AWS_PROFILE": ""},
    )
    assert result.returncode == 0, result.stderr
    assert "plan_complete=no_resources_changed" in result.stdout
    assert "template_static_checks=ok" in result.stdout


def test_cloudformation_template_static_shape() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    for marker in (
        "AWS::SageMaker::Model",
        "AWS::SageMaker::EndpointConfig",
        "AWS::SageMaker::Endpoint",
        "EnableNetworkIsolation: true",
        "AWS::IAM::Role",
        "AWS::Logs::LogGroup",
        "CostCenter",
        "DeletionPolicy: Retain",
        "ImageUri",
        "ModelDataUrl",
        "InstanceType",
        "CreateTrainingJob",
    ):
        assert marker in text
    assert re.search(r"AKIA[0-9A-Z]{16}", text) is None


def test_server_module_import_smoke() -> None:
    # Import without constructing a torch runtime or loading a bundle.
    from distillery_inference import server
    from distillery_inference.schemas import InferRequest

    assert hasattr(server, "create_app")
    request = InferRequest(
        model_id="model_sequence_kd",
        artifact_id="artifact_sequence_kd",
        task="transaction_review",
        example_id=None,
        input={"amount_minor": 1},
    )
    assert request.model_id.startswith("model_")


def test_requirements_serve_pins_uvicorn() -> None:
    requirements = (
        ROOT / "containers" / "inference" / "requirements-serve.txt"
    ).read_text(encoding="utf-8")
    assert "uvicorn==0.32.1" in requirements


@pytest.mark.parametrize(
    "script",
    [BUILD_SCRIPT, PUBLISH_SCRIPT, DEPLOY_SCRIPT],
)
def test_shell_scripts_are_executable_syntax_clean(script: Path) -> None:
    assert script.is_file()
    # bash -n is the portable syntax check; shellcheck is used when installed.
    syntax = subprocess.run(
        ["bash", "-n", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr
    shellcheck = subprocess.run(
        ["shellcheck", "-x", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    if shellcheck.returncode == 127:
        pytest.skip("shellcheck not installed")
    assert shellcheck.returncode == 0, shellcheck.stdout + shellcheck.stderr


def test_ml_compatibility_json_is_reused_not_forked() -> None:
    # Inference image must reference training pins; do not duplicate a divergent copy.
    inference_dir = ROOT / "containers" / "inference"
    assert not (inference_dir / "ml-compatibility.json").exists()
    training = json.loads(
        (ROOT / "containers" / "training" / "ml-compatibility.json").read_text(
            encoding="utf-8"
        )
    )
    assert training["torch_version"] == "2.4.1"
    assert "0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726" in training[
        "base_image"
    ]
