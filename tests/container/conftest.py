"""Shared paths for training-container static tests."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO / "containers" / "training" / "Dockerfile"
DOCKERIGNORE = REPO / "containers" / "training" / ".dockerignore"
BUILD_MANIFEST_SCHEMA = REPO / "containers" / "training" / "build-manifest.schema.json"
DOWNSTREAM_CONTRACT = REPO / "containers" / "training" / "downstream_contract.json"
ECR_TEMPLATE = REPO / "infra" / "ecr" / "template.yaml"
ECR_DEPLOY = REPO / "infra" / "ecr" / "deploy.sh"
BUILD_SCRIPT = REPO / "scripts" / "container" / "build_training_image.sh"
PUBLISH_SCRIPT = REPO / "scripts" / "container" / "publish_training_image.sh"
MANIFEST_TOOL = REPO / "scripts" / "container" / "manifest_tool.py"
STAGE_TOOL = REPO / "scripts" / "container" / "stage_context.py"
ENTRYPOINT = REPO / "containers" / "training" / "container_entrypoint.py"
VERIFY_ML = REPO / "containers" / "training" / "verify_ml_compatibility.py"
ML_COMPATIBILITY = REPO / "containers" / "training" / "ml-compatibility.json"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ecr_template_text() -> str:
    return ECR_TEMPLATE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def build_script_text() -> str:
    return BUILD_SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def publish_script_text() -> str:
    return PUBLISH_SCRIPT.read_text(encoding="utf-8")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_manifest(
    *,
    config_id: str | None = "sha256:" + ("1" * 64),
    verified: bool = False,
    compatibility: str = "compatible",
) -> dict[str, Any]:
    digest = "sha256:" + ("2" * 64)
    repository_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/distillery-training"
    registry = {
        "repository_uri": repository_uri if verified else None,
        "image_digest": digest if verified else None,
        "digest_uri": f"{repository_uri}@{digest}" if verified else None,
        "verified": verified,
        "verified_at": (datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if verified else None),
        "scan_status": "COMPLETE" if verified else None,
        "critical_findings": 0 if verified else None,
        "high_findings": 0 if verified else None,
    }
    return {
        "schema_version": "distillery.training.image.v2",
        "image_name": "distillery-training",
        "repository_name": "distillery-training",
        "source": {
            "commit_sha": "a" * 40,
            "reviewed_commit_sha": "a" * 40,
            "tree_sha256": "b" * 64,
            "clean": True,
            "commit_bound": True,
        },
        "package_lock_sha256": "c" * 64,
        "base": {
            "reference": (
                "pytorch/pytorch@sha256:"
                "0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726"
            ),
            "digest": ("sha256:0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726"),
            "torch_version": "2.4.1",
            "cuda_version": "12.4",
            "cudnn_major": 9,
        },
        "ml_compatibility": {
            "status": compatibility,
            "expected": {
                "torch_version": "2.4.1",
                "torch_cuda_version": "12.4",
                "cudnn_major": 9,
                "bitsandbytes_version": "0.44.1",
                "base_provided_packages": [
                    "nvidia-cublas-cu12",
                    "nvidia-cuda-cupti-cu12",
                    "nvidia-cuda-nvrtc-cu12",
                    "nvidia-cuda-runtime-cu12",
                    "nvidia-cudnn-cu12",
                    "nvidia-cufft-cu12",
                    "nvidia-curand-cu12",
                    "nvidia-cusolver-cu12",
                    "nvidia-cusparse-cu12",
                    "nvidia-nccl-cu12",
                    "nvidia-nvjitlink-cu12",
                    "nvidia-nvtx-cu12",
                    "torch",
                    "triton",
                ],
            },
            "actual": {
                "torch_version": "2.4.1",
                "bitsandbytes_version": "0.44.1",
                "forbidden_packages": [],
                "base_provided_packages": ["torch"],
                "unexpected_accelerator_packages": [],
            },
            "reasons": [] if compatibility == "compatible" else ["blocked by lock"],
        },
        "tag": "pinned-training-aaaaaaaaaaaa-cccccccccccc",
        "local": {
            "config_id": config_id,
        },
        "registry": registry,
        "dry_run": config_id is None,
        "created_at": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
