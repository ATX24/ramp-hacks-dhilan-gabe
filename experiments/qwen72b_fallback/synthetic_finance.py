"""Synthetic finance examples for 72B oracle/sequence-SFT (no customer data)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from distillery.contracts.hashing import content_sha256

TaskName = Literal[
    "transaction_review",
    "variance_analysis",
    "merchant_tagging",
    "cash_reconciliation",
]


class SyntheticFinanceExample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str = Field(min_length=1)
    task: TaskName
    prompt_text: str = Field(min_length=1)
    oracle_completion: str = Field(min_length=1)
    synthetic: Literal[True] = True
    contains_customer_data: Literal[False] = False
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _bind(self) -> SyntheticFinanceExample:
        expected = example_sha256(
            example_id=self.example_id,
            task=self.task,
            prompt_text=self.prompt_text,
            oracle_completion=self.oracle_completion,
        )
        if self.record_sha256 != expected:
            raise ValueError(f"record_sha256 mismatch for {self.example_id}")
        lowered = (self.prompt_text + " " + self.oracle_completion).lower()
        for banned in ("ssn", "social security", "real customer", "live bank account"):
            if banned in lowered:
                raise ValueError(f"banned real-data phrase {banned!r} in synthetic example")
        return self


def example_sha256(
    *,
    example_id: str,
    task: str,
    prompt_text: str,
    oracle_completion: str,
) -> str:
    return content_sha256(
        {
            "example_id": example_id,
            "task": task,
            "prompt_text": prompt_text,
            "oracle_completion": oracle_completion,
            "synthetic": True,
        }
    )


def build_example(
    *,
    example_id: str,
    task: TaskName,
    prompt_text: str,
    oracle_completion: str,
) -> SyntheticFinanceExample:
    digest = example_sha256(
        example_id=example_id,
        task=task,
        prompt_text=prompt_text,
        oracle_completion=oracle_completion,
    )
    return SyntheticFinanceExample(
        example_id=example_id,
        task=task,
        prompt_text=prompt_text,
        oracle_completion=oracle_completion,
        record_sha256=digest,
    )


def rehearsal_corpus() -> list[SyntheticFinanceExample]:
    """24 synthetic examples (3 updates × global batch 8)."""
    templates: list[tuple[TaskName, str, str]] = [
        (
            "transaction_review",
            "Review synthetic card swipe {n}: merchant=ACME OFFICE SUPPLY amount=42.15 USD. "
            "Flag policy issues if any.",
            "No policy violation. Office supply under $50 threshold; approve.",
        ),
        (
            "variance_analysis",
            "Budget line Travel for cost center CC-{n} planned=10000 actual=11250. "
            "Explain variance.",
            "Travel is 12.5% over plan (+1250). Likely fare spike; request receipt pack.",
        ),
        (
            "merchant_tagging",
            "Tag synthetic merchant string 'UBER TRIP HELP {n}'.",
            "category=ground_transport; subcategory=rideshare; confidence=high.",
        ),
        (
            "cash_reconciliation",
            "Reconcile synthetic cash drawer #{n}: system=500.00 counted=498.50.",
            "Short 1.50 USD. Record as cash_over_short; investigate till variance.",
        ),
    ]
    rows: list[SyntheticFinanceExample] = []
    for index in range(24):
        task, prompt_t, completion_t = templates[index % len(templates)]
        rows.append(
            build_example(
                example_id=f"syn_finance_{index:04d}",
                task=task,
                prompt_text=prompt_t.format(n=index),
                oracle_completion=completion_t,
            )
        )
    return rows


def corpus_sha256(rows: list[SyntheticFinanceExample]) -> str:
    ordered = sorted(rows, key=lambda row: row.example_id)
    return content_sha256([row.model_dump(mode="json") for row in ordered])


def precompute_trajectory_stub(rows: list[SyntheticFinanceExample]) -> dict[str, Any]:
    """Offline teacher/tool trajectory placeholder excluded from warm timer."""
    return {
        "schema_version": "distillery.qwen72b_fallback.precomputed_trajectories.v1",
        "runtime": "offline_precomputed_only",
        "included_in_warm_timer": False,
        "example_ids": [row.example_id for row in sorted(rows, key=lambda r: r.example_id)],
        "tool_traces": [],
        "notes": (
            "Tool/teacher trajectories are generated before CreateTrainingJob warm "
            "time; training consumes sealed oracle sequences only."
        ),
        "corpus_sha256": corpus_sha256(rows),
    }
