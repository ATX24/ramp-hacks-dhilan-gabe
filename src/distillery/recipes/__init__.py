"""Distillery recipe implementations (sequence.v1, logit.v1, auto resolver)."""

from __future__ import annotations

from distillery.recipes.auto import (
    AutoResolutionRecord,
    build_auto_input_from_flags,
    require_trainable_resolution,
    resolve_recipe,
)
from distillery.recipes.base import (
    CompletionMask,
    JointTokenizationEvidence,
    MaterializationReport,
    MaterializedExample,
    Recipe,
    RecipeContext,
    RecipeMode,
    ResponseRecord,
    is_pinned_revision,
    require_pinned_revision,
)
from distillery.recipes.logit_v1 import (
    LogitV1Config,
    LogitV1Recipe,
    MemoryDryRunEvidence,
    assert_frozen_teacher,
    assert_matched_ce_ablation,
    assert_tokenizer_compatible,
    compare_matched_ce_ablation_manifests,
    memory_dry_run_evidence_sha256,
    validate_memory_dry_run_evidence,
)
from distillery.recipes.sequence_v1 import (
    IGNORE_INDEX,
    SequenceV1Config,
    SequenceV1Recipe,
    build_completion_only_mask,
    build_completion_only_mask_from_joint,
    materialize_sequence_examples,
    retokenize_text_pair,
    validate_response_text,
)

__all__ = [
    "IGNORE_INDEX",
    "AutoResolutionRecord",
    "CompletionMask",
    "JointTokenizationEvidence",
    "LogitV1Config",
    "LogitV1Recipe",
    "MemoryDryRunEvidence",
    "MaterializationReport",
    "MaterializedExample",
    "Recipe",
    "RecipeContext",
    "RecipeMode",
    "ResponseRecord",
    "SequenceV1Config",
    "SequenceV1Recipe",
    "assert_frozen_teacher",
    "assert_matched_ce_ablation",
    "assert_tokenizer_compatible",
    "build_auto_input_from_flags",
    "build_completion_only_mask",
    "build_completion_only_mask_from_joint",
    "compare_matched_ce_ablation_manifests",
    "is_pinned_revision",
    "materialize_sequence_examples",
    "memory_dry_run_evidence_sha256",
    "require_trainable_resolution",
    "require_pinned_revision",
    "resolve_recipe",
    "retokenize_text_pair",
    "validate_response_text",
    "validate_memory_dry_run_evidence",
]
