"""Session-scoped sealed corpora for data quality tests."""

from __future__ import annotations

import pytest

from distillery.data.generate import CORPUS_FULL, CORPUS_SMOKE, generate_corpus


@pytest.fixture(scope="session")
def smoke_corpus():
    return generate_corpus(CORPUS_SMOKE, check_near_duplicates=True)


@pytest.fixture(scope="session")
def full_corpus():
    return generate_corpus(CORPUS_FULL, check_near_duplicates=True)
