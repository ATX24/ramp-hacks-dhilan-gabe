"""Inference test bootstrap: package path + offline env defaults."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_APP = ROOT / "apps" / "inference"

if str(INFERENCE_APP) not in sys.path:
    sys.path.insert(0, str(INFERENCE_APP))

# Tests use FakeRuntime and never download models.
os.environ.setdefault("DISTILLERY_INFERENCE_RUNTIME", "fake")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


@pytest.fixture
def repo_root() -> Path:
    return ROOT
