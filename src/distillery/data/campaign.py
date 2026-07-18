"""Campaign manifest builders for finance_world.v1 / finance_world.v2.

These helpers only seal dataset/campaign metadata. They do not launch training,
touch AWS, or mutate active UI/API/container paths.
"""

from __future__ import annotations

from typing import Any, Literal

from distillery.contracts.hashing import content_sha256
from distillery.data.generate import (
    CORPUS_FULL,
    CORPUS_FULL_V2,
    CORPUS_SMOKE,
    CORPUS_SMOKE_V2,
    CorpusSpec,
    GeneratedCorpus,
    generate_corpus,
)
from distillery.data.mixture import TASK_MIXTURE, TASK_MIXTURE_V2
from distillery.proof.protocol_v2 import (
    PROOF_PROTOCOL_ID_V1,
    PROOF_PROTOCOL_ID_V2,
    finance_proof_v2_document,
    finance_proof_v2_sha256,
)
from distillery.training.batching import DEFAULT_FINANCE_MIXTURE, FINANCE_MIXTURE_V2

CampaignWorld = Literal["finance_world.v1", "finance_world.v2"]
CampaignCorpus = Literal["smoke", "full"]


def campaign_corpus_spec(
    world: CampaignWorld,
    corpus: CampaignCorpus,
) -> CorpusSpec:
    if world == "finance_world.v1":
        return CORPUS_SMOKE if corpus == "smoke" else CORPUS_FULL
    if world == "finance_world.v2":
        return CORPUS_SMOKE_V2 if corpus == "smoke" else CORPUS_FULL_V2
    raise ValueError(f"unsupported campaign world {world!r}")


def build_campaign_manifest(
    *,
    world: CampaignWorld,
    corpus: CampaignCorpus = "smoke",
    campaign_id: str,
    shared_artifact_id: str = "tinyfable_generalist",
    generated: GeneratedCorpus | None = None,
    check_near_duplicates: bool = True,
) -> dict[str, Any]:
    """Build a sealed campaign manifest for a shared-generalist TinyFable run."""
    spec = campaign_corpus_spec(world, corpus)
    corpus_obj = generated or generate_corpus(
        spec,
        check_near_duplicates=check_near_duplicates,
    )
    if world == "finance_world.v1":
        protocol_id = PROOF_PROTOCOL_ID_V1
        protocol_doc = {
            "id": PROOF_PROTOCOL_ID_V1,
            "finance_world": "finance_world.v1",
            "task_mixture": {task.value: weight for task, weight in TASK_MIXTURE.items()},
            "primary_tasks": ["transaction_review", "variance_analysis"],
            "diagnostic_tasks": ["cash_reconciliation"],
            "specialist_routing": False,
            "shared_model": shared_artifact_id,
        }
        protocol_sha = content_sha256(protocol_doc)
        sampler_mixture = dict(DEFAULT_FINANCE_MIXTURE.task_weights)
    else:
        protocol_id = PROOF_PROTOCOL_ID_V2
        protocol_doc = finance_proof_v2_document()
        protocol_sha = finance_proof_v2_sha256()
        sampler_mixture = dict(FINANCE_MIXTURE_V2.task_weights)

    merchant_count = sum(
        1 for example in corpus_obj.examples if example.task.value == "merchant_tagging"
    )
    manifest: dict[str, Any] = {
        "schema_version": "distillery.campaign_manifest.v1",
        "campaign_id": campaign_id,
        "finance_world": world,
        "corpus": spec.name,
        "shared_artifact_id": shared_artifact_id,
        "specialist_routing": False,
        "routing_policy": "same_artifact_all_primary_tasks",
        "proof_protocol": {
            "id": protocol_id,
            "sha256": protocol_sha,
            "document": protocol_doc,
        },
        "dataset": {
            "envelope_schema_version": corpus_obj.spec.schema_version,
            "generator_revision": corpus_obj.spec.generator_revision,
            "total_examples": corpus_obj.manifest["total_examples"],
            "content_sha256": corpus_obj.manifest["content_sha256"],
            "split_sha256": corpus_obj.manifest["split_sha256"],
            "order_sha256": corpus_obj.manifest["order_sha256"],
            "task_mixture_target": {
                task.value: weight for task, weight in corpus_obj.spec.task_mixture.items()
            },
            "merchant_tagging_examples": merchant_count,
        },
        "sampler_mixture": sampler_mixture,
        "difficulty_mixture": dict(DEFAULT_FINANCE_MIXTURE.difficulty_weights),
        "trainable_arms": [
            "oracle_sft",
            "sequence_kd",
            "logit_kd",
            "ce_ablation",
        ],
        "notes": (
            "One shared TinyFable generalist artifact; never train per-task specialists."
        ),
    }
    if world == "finance_world.v2":
        manifest["dataset"]["min_full_merchant_examples"] = (
            1000 if corpus == "full" else 0
        )
        manifest["dataset"]["task_mixture_declared"] = {
            task.value: weight for task, weight in TASK_MIXTURE_V2.items()
        }
    manifest["manifest_sha256"] = content_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest
