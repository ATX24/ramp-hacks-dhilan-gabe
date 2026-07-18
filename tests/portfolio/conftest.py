"""Path setup for experiment-package tests."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tests.portfolio.support import make_plan  # noqa: E402


@pytest.fixture
def plan():
    return make_plan()
