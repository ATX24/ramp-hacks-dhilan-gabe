"""Shared fixtures for proof tests."""

from __future__ import annotations

from typing import Any

import pytest

from distillery.proof.testing import txn_gold, var_gold


@pytest.fixture
def perfect_txn() -> dict[str, Any]:
    return txn_gold()


@pytest.fixture
def perfect_var() -> dict[str, Any]:
    return var_gold()
