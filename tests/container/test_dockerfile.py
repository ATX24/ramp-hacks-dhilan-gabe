"""Dockerfile instruction-level packaging and security tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

from conftest import DOCKERFILE, DOCKERIGNORE, ML_COMPATIBILITY

BASE_REFERENCE = (
    "pytorch/pytorch@sha256:0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726"
)
SECRET_RE = re.compile(
    r"(AKIA[0-9A-Z]{16}|aws_secret_access_key\s*=\s*\S+|"
    r"HF_TOKEN\s*=\s*hf_|AWS_SECRET_ACCESS_KEY\s*=\s*\S+)",
    re.I,
)


def docker_instructions(path: Path) -> list[tuple[str, str]]:
    logical_lines: list[str] = []
    current = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        current = f"{current} {stripped}".strip()
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        logical_lines.append(current)
        current = ""
    if current:
        logical_lines.append(current)
    return [
        (line.split(maxsplit=1)[0].upper(), line.split(maxsplit=1)[1])
        for line in logical_lines
        if len(line.split(maxsplit=1)) == 2
    ]


def test_base_and_build_metadata_are_immutable() -> None:
    instructions = docker_instructions(DOCKERFILE)
    args = [value for instruction, value in instructions if instruction == "ARG"]
    assert f"BASE_IMAGE={BASE_REFERENCE}" in args
    assert ("FROM", "${BASE_IMAGE}") in instructions
    labels = " ".join(value for instruction, value in instructions if instruction == "LABEL")
    for label in (
        "distillery.source.sha=",
        "distillery.source.tree.sha256=",
        "distillery.package.lock.sha256=",
        "distillery.base.image=",
        'distillery.ml.torch="2.4.1"',
        'distillery.ml.cuda="12.4"',
        'distillery.ml.bitsandbytes="0.44.1"',
        'distillery.qwen72b.trainer="experiments.qwen72b_fallback.train"',
        'distillery.qwen72b.attention.backend="sdpa_math"',
        'distillery.qwen72b.flash_attention_2="false"',
        'distillery.runtime.uid="1000"',
    ):
        assert label in labels
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "SOURCE_DATE_EPOCH=0" in text
    assert "UV_ARCHIVE_SHA256=" in text
    assert "uv archive checksum mismatch" in text
    assert "apt-get" not in text


def test_package_metadata_and_runtime_files_are_copied() -> None:
    instructions = docker_instructions(DOCKERFILE)
    copies = [value for instruction, value in instructions if instruction == "COPY"]
    copy_text = "\n".join(copies)
    assert "pyproject.toml uv.lock README.md LICENSE" in copy_text
    assert "src/distillery ./src/distillery" in copy_text
    assert "experiments ./experiments" in copy_text
    assert "container_entrypoint.py" in copy_text
    assert "ml-compatibility.json" in copy_text
    assert "verify_ml_compatibility.py" in copy_text
    assert all("--chown=root:root" in copy for copy in copies)
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert 'PYTHONPATH="/opt/distillery/src:/opt/distillery"' in text
    assert "python -m experiments.aws_smoke.train --help" in text
    assert "python -m experiments.qwen72b_fallback.train --help" in text


def test_lock_and_runtime_checks_surround_uv_sync() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    lock_check = text.index("verify_ml_compatibility.py lock")
    sync = text.index("uv sync --frozen --no-dev --extra ml --no-editable")
    post_check = text.index(
        "verify_ml_compatibility.py runtime",
        sync,
    )
    assert lock_check < sync < post_check
    assert "UV_PROJECT_ENVIRONMENT=/opt/distillery/.venv" in text
    assert "python -m venv --system-site-packages" in text
    assert "--require-bitsandbytes" in text
    assert not re.search(r"(pip|uv pip) install\s+torch", text)
    assert "cu13" not in text
    compatibility = json.loads(ML_COMPATIBILITY.read_text(encoding="utf-8"))
    for package in compatibility["base_provided_packages"]:
        assert f"--no-install-package {package}" in text


def test_entrypoint_is_signal_aware_root_init_then_non_root_runtime() -> None:
    instructions = docker_instructions(DOCKERFILE)
    assert ("STOPSIGNAL", "SIGTERM") in instructions
    assert (
        "ENTRYPOINT",
        '["python", "/opt/distillery/container_entrypoint.py"]',
    ) in instructions
    healthcheck = next(value for instruction, value in instructions if instruction == "HEALTHCHECK")
    assert 'CMD ["python", "/opt/distillery/container_entrypoint.py", "--health"]' in healthcheck
    cmd = next(value for instruction, value in instructions if instruction == "CMD")
    for required in (
        "--manifest",
        "/opt/ml/input/data/manifest/manifest.json",
        "--responses",
        "/opt/ml/input/data/responses/responses.jsonl",
        "--output-dir",
        "/opt/ml/model/validation",
        "--validate-only",
    ):
        assert required in cmd
    assert 'distillery.runtime.uid="1000"' in DOCKERFILE.read_text(encoding="utf-8")


def test_offline_model_materialization_conventions() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    for marker in (
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "HF_DATASETS_OFFLINE=1",
        "DISTILLERY_REQUIRE_PINNED_REVISION=1",
        "DISTILLERY_SAGEMAKER_MODEL_INPUT=/opt/ml/input/data/models",
        "DISTILLERY_SAGEMAKER_RESPONSES_INPUT=/opt/ml/input/data/responses",
    ):
        assert marker in text
    directory_setup = text[text.index("mkdir -p") : text.index("chown -R")]
    for mounted_channel in (
        "/opt/ml/input/data/manifest",
        "/opt/ml/input/data/responses",
        "/opt/ml/input/data/dataset",
        "/opt/ml/input/data/models",
    ):
        assert mounted_channel not in directory_setup


def test_no_credentials_or_weights_can_enter_context() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    ignore = DOCKERIGNORE.read_text(encoding="utf-8")
    assert SECRET_RE.search(dockerfile) is None
    assert "private_key_markers" in dockerfile
    assert ignore.splitlines()[3].strip() == "**"
    allowed = {
        line[2:] for line in ignore.splitlines() if line.startswith("!/") and not line.endswith("/")
    }
    assert allowed == {
        ".dockerignore",
        "SOURCE_FILES.json",
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
        "src/distillery/**",
        "experiments/__init__.py",
        "experiments/aws_smoke/__init__.py",
        "experiments/aws_smoke/artifacts.py",
        "experiments/aws_smoke/channels.py",
        "experiments/aws_smoke/deadline.py",
        "experiments/aws_smoke/device_mapping.py",
        "experiments/aws_smoke/loss_wiring.py",
        "experiments/aws_smoke/manifests.py",
        "experiments/aws_smoke/memory.py",
        "experiments/aws_smoke/model_evidence.py",
        "experiments/aws_smoke/pins.py",
        "experiments/aws_smoke/profile.py",
        "experiments/aws_smoke/tokenization.py",
        "experiments/aws_smoke/train.py",
        "experiments/qwen72b_fallback/**",
        "containers/training/Dockerfile",
        "containers/training/container_entrypoint.py",
        "containers/training/ml-compatibility.json",
        "containers/training/verify_ml_compatibility.py",
    }
