"""Sealed-manifest recipe and dataset-reference invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from distillery.contracts.manifest import ManifestDatasetRef, ManifestRecipe
from distillery.contracts.recipes import (
    AUTO_BASELINE_PRECEDENCE_REASON,
    AUTO_LOGIT_REASONS,
    AUTO_SEQUENCE_RESPONSES_REASONS,
)

HEX64 = "a" * 64


@pytest.mark.parametrize(
    ("requested", "resolved"),
    [
        ("sequence.v1", "logit.v1"),
        ("sequence.v1", "do_not_distill"),
        ("logit.v1", "sequence.v1"),
        ("logit.v1", "do_not_distill"),
    ],
)
def test_explicit_manifest_recipe_cannot_silently_downgrade(
    requested: str,
    resolved: str,
) -> None:
    with pytest.raises(ValidationError, match="must resolve only to"):
        ManifestRecipe(
            requested=requested,
            resolved=resolved,
            resolver_reasons=("explicit_request",),
        )


@pytest.mark.parametrize("requested", ["sequence.v1", "logit.v1"])
def test_explicit_manifest_recipe_requires_explicit_reason(requested: str) -> None:
    with pytest.raises(ValidationError, match="explicit recipes require"):
        ManifestRecipe(
            requested=requested,
            resolved=requested,
            resolver_reasons=("usable_responses_present",),
        )


@pytest.mark.parametrize(
    ("resolved", "reasons"),
    [
        ("sequence.v1", AUTO_SEQUENCE_RESPONSES_REASONS),
        ("logit.v1", AUTO_LOGIT_REASONS),
        ("do_not_distill", (AUTO_BASELINE_PRECEDENCE_REASON,)),
    ],
)
def test_auto_manifest_resolution_is_auditable(
    resolved: str,
    reasons: tuple[str, ...],
) -> None:
    recipe = ManifestRecipe(
        requested="auto",
        resolved=resolved,
        resolver_reasons=reasons,
    )
    assert recipe.resolved == resolved


def test_auto_manifest_requires_reasons() -> None:
    with pytest.raises(ValidationError, match="at least one resolver reason"):
        ManifestRecipe(requested="auto", resolved="sequence.v1")


def test_auto_do_not_distill_requires_locked_baseline_reason() -> None:
    with pytest.raises(ValidationError, match="cheaper-baseline precedence"):
        ManifestRecipe(
            requested="auto",
            resolved="do_not_distill",
            resolver_reasons=("arbitrary_reason",),
        )


def test_manifest_dataset_ref_requires_train_and_validation() -> None:
    for split_hashes in (
        {},
        {"train": HEX64},
        {"validation": HEX64},
    ):
        with pytest.raises(ValidationError, match="missing required splits"):
            ManifestDatasetRef(
                dataset_id="ds_finance_world_v1",
                uri="s3://bucket/dataset/",
                sha256=HEX64,
                split_sha256=split_hashes,
            )


@pytest.mark.parametrize("split", ["train", "validation", "iid_test", "ood_test", "test"])
def test_manifest_validates_every_split_hash(split: str) -> None:
    split_hashes = {"train": HEX64, "validation": HEX64}
    split_hashes[split] = "not-a-sha256"
    with pytest.raises(ValidationError):
        ManifestDatasetRef(
            dataset_id="ds_finance_world_v1",
            uri="s3://bucket/dataset/",
            sha256=HEX64,
            split_sha256=split_hashes,
        )


def test_manifest_rejects_unknown_split_name() -> None:
    with pytest.raises(ValidationError):
        ManifestDatasetRef(
            dataset_id="ds_finance_world_v1",
            uri="s3://bucket/dataset/",
            sha256=HEX64,
            split_sha256={
                "train": HEX64,
                "validation": HEX64,
                "shadow": HEX64,
            },
        )
