"""Curate → Synthesize → Train-planned → Prove without training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fakes import FakeDistillery, FakeTrainingLaunchError


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_fake_pipeline_plan_only_never_trains(golden_jsonl: Path) -> None:
    client = FakeDistillery(allow_train=False)

    # Curate
    dataset = client.datasets.create(golden_jsonl)
    assert dataset.dataset_id.startswith("ds_")
    assert dataset.example_count >= 1
    assert len(dataset.content_sha256) == 64

    # Synthesize inventory
    rows = dataset.rows
    label_counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("provenance", {}).get("label_source", "unknown"))
        label_counts[source] = label_counts.get(source, 0) + 1
    assert sum(label_counts.values()) == dataset.example_count
    assert label_counts.get("oracle", 0) > 0

    # Train-planned (plan only)
    plan = client.plan_distillation(dataset, recipe="auto")
    assert client.plan_calls == 1
    assert client.distill_calls == 0
    assert plan.launched_training is False
    assert plan.as_dict()["launches_training"] is False
    assert plan.resolved_recipe in {"sequence.v1", "logit.v1", "do_not_distill", None}

    # Prove placeholder from plan
    report = client.prove_from_plan(plan)
    assert report.proof_status in {
        "insufficient_evidence",
        "do_not_distill",
        "proved",
        "failed_quality",
        "failed_economics",
    }
    assert "No training was executed." in report.limitations


def test_fake_distill_blocked_without_ack_and_allow_train(golden_jsonl: Path) -> None:
    client = FakeDistillery(allow_train=False)
    dataset = client.create_dataset(golden_jsonl)
    with pytest.raises(FakeTrainingLaunchError):
        client.distill(dataset, recipe="auto", training_acknowledged=False)
    with pytest.raises(FakeTrainingLaunchError):
        client.distill(dataset, recipe="auto", training_acknowledged=True)
    assert client.distill_calls == 2


def test_example_run_pipeline_plan_mode(golden_jsonl: Path) -> None:
    from finance_generalist import LocalProtocolAdapter, run_pipeline

    result = run_pipeline(
        dataset_path=golden_jsonl,
        mode="plan",
        recipe="auto",
        seed=17,
        client=LocalProtocolAdapter(),
        acknowledge_training=False,
    )
    stages = result["stages"]
    assert set(stages) == {"curate", "synthesize", "train", "prove"}
    assert stages["train"]["training_launched"] is False
    assert stages["train"]["plan"]["launches_training"] is False
    assert result["claims"]["benchmarks"] == "pending"
    assert stages["prove"]["proof_status"] in {
        "insufficient_evidence",
        "do_not_distill",
    }


def test_example_cli_plan_exits_zero(golden_jsonl: Path) -> None:
    from finance_generalist import main

    code = main(["--dataset", str(golden_jsonl), "--mode", "plan", "--json"])
    assert code == 0


def test_example_cli_train_without_ack_fails(golden_jsonl: Path) -> None:
    from finance_generalist import main

    code = main(["--dataset", str(golden_jsonl), "--mode", "train"])
    assert code == 3


def test_contracts_auto_resolver_available_or_skipped() -> None:
    """Document integration: contracts resolver preferred when importable."""
    pytest.importorskip("distillery.contracts.recipes")
    from distillery.contracts.recipes import AutoResolverInput, resolve_requested_recipe

    result = resolve_requested_recipe(
        "auto",
        auto_input=AutoResolverInput(usable_responses_exist=True),
    )
    assert result.resolved == "sequence.v1"
