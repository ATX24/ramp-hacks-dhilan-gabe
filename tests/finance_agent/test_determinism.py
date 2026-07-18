"""Determinism, corpus sizes, and separated materialization checks."""

from __future__ import annotations

import json

from distillery.contracts.tasks import SplitName
from distillery.finance_agent.generate import (
    CORPUS_PLANNED,
    CORPUS_SMOKE,
    GeneratedAgentCorpus,
    generate_agent_corpus,
)
from distillery.finance_agent.validate import validate_episode
from distillery.finance_agent.world import agent_world_from_payload


def test_smoke_generation_and_all_seals_are_deterministic() -> None:
    first = generate_agent_corpus(CORPUS_SMOKE)
    second = generate_agent_corpus(CORPUS_SMOKE)
    assert first.manifest == second.manifest
    assert first.proof_protocol == second.proof_protocol
    assert [example.episode_sha256 for example in first.examples] == [
        example.episode_sha256 for example in second.examples
    ]


def test_seed_change_changes_corpus() -> None:
    first = generate_agent_corpus(CORPUS_SMOKE, seed=17)
    second = generate_agent_corpus(CORPUS_SMOKE, seed=23)
    assert first.manifest["corpus_sha256"] != second.manifest["corpus_sha256"]
    assert first.manifest["corpus_order_sha256"] != second.manifest["corpus_order_sha256"]


def test_smoke_exact_sizes_and_split_contract(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    assert len(smoke_corpus.examples) == 48
    assert len(smoke_corpus.by_split[SplitName.TRAIN]) == 24
    assert len(smoke_corpus.by_split[SplitName.VALIDATION]) == 8
    assert len(smoke_corpus.by_split[SplitName.TEST]) == 8
    assert len(smoke_corpus.by_split[SplitName.OOD_TEST]) == 8


def test_planned_corpus_materializes_all_2200_rows(
    planned_corpus: GeneratedAgentCorpus,
) -> None:
    assert CORPUS_PLANNED.total_examples == 2_200
    assert len(planned_corpus.examples) == 2_200
    assert planned_corpus.manifest["splits"] == {
        "train": 1_200,
        "validation": 200,
        "iid_test": 400,
        "ood_test": 400,
    }
    assert planned_corpus.leakage.ok


def test_written_corpus_separates_model_gold_and_worlds(
    tmp_path,
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    hashes = smoke_corpus.write(tmp_path)
    assert "model/test.jsonl" in hashes
    assert "gold/test.jsonl" in hashes
    assert "oracle/worlds.jsonl" in hashes
    model_row = json.loads(
        (tmp_path / "model/test.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "gold" not in model_row
    assert "trajectory" not in model_row
    gold_row = json.loads(
        (tmp_path / "gold/test.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "gold" in gold_row

    world_payload = json.loads(
        (tmp_path / "oracle/worlds.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    expected_hash = world_payload.pop("latent_state_hash")
    world = agent_world_from_payload(world_payload)
    assert world.latent_state_hash() == expected_hash
    example = next(item for item in smoke_corpus.examples if item.world_id == world.world_id)
    validate_episode(example, world=world)


def test_manifest_has_honest_null_artifact_economics_and_license_states(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    manifest = smoke_corpus.manifest
    assert manifest["label_source_counts"] == {"oracle": 48, "teacher": 0}
    assert manifest["teacher_rollout_artifact_sha256"] is None
    assert manifest["model_id"] is None
    assert manifest["tokenizer_sha256"] is None
    assert manifest["chat_template_sha256"] is None
    assert manifest["license_disposition"] == "unknown"
    assert manifest["cost_disposition"] == "unknown"
    assert manifest["proof_status"] == "not_ready"
