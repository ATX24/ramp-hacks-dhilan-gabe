"""Shared contract-test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "finance_world_v1"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def golden_lines(fixture_dir: Path) -> list[str]:
    text = (fixture_dir / "golden.jsonl").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 12
    return lines


@pytest.fixture(scope="session")
def golden_records(golden_lines: list[str]) -> list[dict]:
    return [json.loads(line) for line in golden_lines]


@pytest.fixture(scope="session")
def oracle_expected(fixture_dir: Path) -> dict:
    return json.loads((fixture_dir / "oracle_expected.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def fixture_manifest(fixture_dir: Path) -> dict:
    return json.loads((fixture_dir / "fixture_manifest.json").read_text(encoding="utf-8"))
