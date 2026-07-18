"""Finance Agent: conversational finance Q&A over a sandboxed synthetic tool world.

Isolated from TinyFable ``finance_world.*`` task envelopes. See CONTEXT.md.
"""

from __future__ import annotations

from distillery.finance_agent.contracts import (
    SCHEMA_VERSION_FINANCE_AGENT,
    AgentEpisodeEnvelope,
    AgentModelInput,
    CaseFamily,
    ToolName,
)
from distillery.finance_agent.generate import (
    CORPUS_PLANNED,
    CORPUS_SMOKE,
    GeneratedAgentCorpus,
    generate_agent_corpus,
)
from distillery.finance_agent.metrics import AgentMetrics, score_episode
from distillery.finance_agent.proof import FinanceAgentProofProtocol

__all__ = [
    "SCHEMA_VERSION_FINANCE_AGENT",
    "AgentEpisodeEnvelope",
    "AgentMetrics",
    "AgentModelInput",
    "CORPUS_PLANNED",
    "CORPUS_SMOKE",
    "CaseFamily",
    "FinanceAgentProofProtocol",
    "GeneratedAgentCorpus",
    "ToolName",
    "generate_agent_corpus",
    "score_episode",
]
