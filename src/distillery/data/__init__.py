"""Synthetic finance world: latent oracle, renderers, corpus generation, leakage checks."""

from __future__ import annotations

from distillery.data.generate import (
    CORPUS_FULL,
    CORPUS_SMOKE,
    CorpusSpec,
    GeneratedCorpus,
    generate_corpus,
)
from distillery.data.leakage import LeakageReport, check_leakage
from distillery.data.oracle import GENERATOR_REVISION, solve_task
from distillery.data.validate import ValidationResult, validate_example, validate_output
from distillery.data.world import LatentWorld, build_world

__all__ = [
    "CORPUS_FULL",
    "CORPUS_SMOKE",
    "CorpusSpec",
    "GENERATOR_REVISION",
    "GeneratedCorpus",
    "LatentWorld",
    "LeakageReport",
    "ValidationResult",
    "build_world",
    "check_leakage",
    "generate_corpus",
    "solve_task",
    "validate_example",
    "validate_output",
]
