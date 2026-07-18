"""E2E fixtures: paths and sys.path wiring for examples/ helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
E2E_DIR = Path(__file__).resolve().parent
GOLDEN_JSONL = REPO_ROOT / "tests" / "fixtures" / "finance_world_v1" / "golden.jsonl"
FIXTURE_MANIFEST = (
    REPO_ROOT / "tests" / "fixtures" / "finance_world_v1" / "fixture_manifest.json"
)

# Path setup must run at import time so test modules can import examples/ and fakes.
for _path in (EXAMPLES_DIR, E2E_DIR):
    _as_str = str(_path)
    if _as_str not in sys.path:
        sys.path.insert(0, _as_str)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def golden_jsonl() -> Path:
    assert GOLDEN_JSONL.is_file(), f"missing fixture: {GOLDEN_JSONL}"
    return GOLDEN_JSONL


@pytest.fixture
def fixture_manifest() -> Path:
    assert FIXTURE_MANIFEST.is_file(), f"missing fixture: {FIXTURE_MANIFEST}"
    return FIXTURE_MANIFEST
