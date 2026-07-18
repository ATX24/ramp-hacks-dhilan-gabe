"""Shared fixtures for finance-world data tests."""

from __future__ import annotations

import pytest

from distillery.data.generate import CORPUS_SMOKE, generate_corpus


@pytest.fixture(scope="module")
def smoke_corpus():
    return generate_corpus(CORPUS_SMOKE, check_near_duplicates=True)
