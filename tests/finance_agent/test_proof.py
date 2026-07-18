"""Sealed Finance Agent proof protocol and paired-evaluation gates."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from distillery.finance_agent.generate import GeneratedAgentCorpus
from distillery.finance_agent.proof import (
    CostDisposition,
    FinanceAgentProofBindings,
    FinanceAgentProofProtocol,
    LicenseDisposition,
    PairedPrediction,
    validate_paired_evaluation,
)


def test_generated_protocol_is_sealed_and_honestly_not_ready(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    protocol = smoke_corpus.proof_protocol
    assert protocol.protocol_sha256
    assert protocol.paired_evaluation is True
    assert protocol.proof_status == "not_ready"
    assert "missing_model_id" in protocol.readiness_blockers
    assert "license_not_approved" in protocol.readiness_blockers
    assert "cost_not_measured" in protocol.readiness_blockers


def test_proof_can_be_ready_only_with_all_exact_bindings() -> None:
    bindings = FinanceAgentProofBindings(
        seed=17,
        corpus_sha256="a" * 64,
        corpus_order_sha256="b" * 64,
        system_prompt_set_sha256="c" * 64,
        tool_schema_set_sha256="d" * 64,
        trajectory_render_template_sha256="e" * 64,
        model_id="model/generalist",
        model_revision="f" * 40,
        model_artifact_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        chat_template_sha256="3" * 64,
        license=LicenseDisposition(
            status="approved",
            license_id="reviewed-license",
            license_text_sha256="4" * 64,
            output_use_reviewed=True,
        ),
        cost=CostDisposition(
            status="measured",
            measurement_artifact_sha256="5" * 64,
            mean_latency_ms=12.5,
            total_cost_usd=1.25,
        ),
    )
    protocol = FinanceAgentProofProtocol.seal(bindings=bindings)
    assert protocol.proof_status == "ready"
    assert protocol.readiness_blockers == ()


def test_unknown_cost_cannot_carry_fabricated_values() -> None:
    with pytest.raises(ValidationError, match="synthetic"):
        CostDisposition(status="unknown", mean_latency_ms=1.0)


def test_paired_evaluation_requires_exact_order_seed_and_input_hash(
    smoke_corpus: GeneratedAgentCorpus,
) -> None:
    examples = smoke_corpus.examples[:2]

    def rows(arm_id: str) -> list[PairedPrediction]:
        return [
            PairedPrediction(
                arm_id=arm_id,
                example_id=example.example_id,
                seed=17,
                model_input_sha256=example.model_input.model_input_sha256,
                trajectory=example.gold.trajectory,
            )
            for example in examples
        ]

    arm_a = rows("arm_a")
    arm_b = rows("arm_b")
    keys = validate_paired_evaluation({"arm_a": arm_a, "arm_b": arm_b})
    assert len(keys) == 2
    with pytest.raises(ValueError, match="paired order"):
        validate_paired_evaluation({"arm_a": arm_a, "arm_b": list(reversed(arm_b))})
