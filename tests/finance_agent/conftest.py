"""Shared immutable generated corpora for Finance Agent tests."""

from __future__ import annotations

import pytest

from distillery.finance_agent.generate import GeneratedAgentCorpus, generate_agent_corpus


@pytest.fixture(scope="session")
def smoke_corpus() -> GeneratedAgentCorpus:
    return generate_agent_corpus("smoke")


@pytest.fixture(scope="session")
def planned_corpus() -> GeneratedAgentCorpus:
    return generate_agent_corpus("planned")
