#!/usr/bin/env python3
"""Finance-generalist Distillery demo (TinyFable path).

Targets the documented high-level SDK surface:

    distillery = Distillery(api_key=...)
    dataset = distillery.datasets.create("./finance_world.jsonl")
    plan = distillery.plan_distillation(dataset, recipe="auto")
    run = distillery.distill(dataset, recipe="auto").wait()  # training only

Defaults to dry-run / plan-only. Training cannot start unless the operator
passes BOTH ``--mode train`` and ``--i-acknowledge-training-will-launch``.

This workstream owns demos only. When ``packages/sdk`` is unavailable, the
script uses an in-process protocol adapter that exercises the same stages
against contracts + fixtures (no SageMaker, no model download, no training).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = REPO_ROOT / "tests" / "fixtures" / "finance_world_v1" / "golden.jsonl"
TRAINING_ACK_FLAG = "--i-acknowledge-training-will-launch"
TRAINING_ACK_PHRASE = "I_ACKNOWLEDGE_THIS_WILL_LAUNCH_TRAINING"

# Ensure sibling example modules import cleanly when run as a script.
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from held_out_selector import DEFAULT_DEMO_SEED, select_held_out  # noqa: E402
from held_out_selector import load_jsonl as load_dataset_jsonl  # noqa: E402


class TrainingSafetyError(RuntimeError):
    """Raised when a call would start training without explicit acknowledgment."""


@dataclass(frozen=True)
class PlanResult:
    """Documented ``plan_distillation()`` shape (subset used by demos)."""

    requested_recipe: str
    resolved_recipe: str | None
    resolver_reasons: tuple[str, ...]
    do_not_distill: bool
    teacher_id: str
    student_id: str
    teacher_revision: str
    student_revision: str
    estimate_low_usd: float
    estimate_high_usd: float
    blockers: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_recipe": self.requested_recipe,
            "resolved_recipe": self.resolved_recipe,
            "resolver_reasons": list(self.resolver_reasons),
            "do_not_distill": self.do_not_distill,
            "teacher": {"id": self.teacher_id, "revision": self.teacher_revision},
            "student": {"id": self.student_id, "revision": self.student_revision},
            "cost_estimate_usd": {
                "low": self.estimate_low_usd,
                "high": self.estimate_high_usd,
            },
            "blockers": list(self.blockers),
            "details": self.details,
            "launches_training": False,
        }


@dataclass(frozen=True)
class DatasetRef:
    dataset_id: str
    uri: str
    content_sha256: str
    example_count: int


@dataclass(frozen=True)
class ProofRef:
    report_id: str
    proof_status: str
    limitations: tuple[str, ...]


class DistilleryClient(Protocol):
    def create_dataset(self, path: Path) -> DatasetRef: ...

    def plan_distillation(
        self, dataset: DatasetRef, *, recipe: str = "auto"
    ) -> PlanResult: ...

    def distill(
        self,
        dataset: DatasetRef,
        *,
        recipe: str = "auto",
        training_acknowledged: bool = False,
    ) -> Any: ...


def try_import_sdk_client(api_key: str | None) -> DistilleryClient | None:
    """Import the real SDK when Workstream 1 has landed; otherwise return None."""
    try:
        from distillery import Distillery  # type: ignore[attr-defined]
    except ImportError:
        try:
            from distillery.client import Distillery  # type: ignore[attr-defined]
        except ImportError:
            return None

    key = api_key or os.environ.get("DISTILLERY_API_KEY")
    if key is None:
        # SDK present but no key: still prefer adapter for local rehearsal.
        return None
    client = Distillery(api_key=key)
    return SdkClientAdapter(client)


class SdkClientAdapter:
    """Thin adapter so the example talks one interface to SDK or fakes."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def create_dataset(self, path: Path) -> DatasetRef:
        datasets = getattr(self._client, "datasets", None)
        if datasets is None or not hasattr(datasets, "create"):
            raise AttributeError("SDK missing datasets.create")
        created = datasets.create(str(path))
        return DatasetRef(
            dataset_id=str(created.dataset_id),
            uri=str(getattr(created, "uri", path.as_uri())),
            content_sha256=str(getattr(created, "content_sha256", "")),
            example_count=int(getattr(created, "example_count", 0)),
        )

    def plan_distillation(self, dataset: DatasetRef, *, recipe: str = "auto") -> PlanResult:
        if hasattr(self._client, "plan_distillation"):
            raw = self._client.plan_distillation(dataset, recipe=recipe)
        else:
            raw = self._client.distillation_runs.plan(dataset, recipe=recipe)
        return _coerce_plan(raw, recipe)

    def distill(
        self,
        dataset: DatasetRef,
        *,
        recipe: str = "auto",
        training_acknowledged: bool = False,
    ) -> Any:
        if not training_acknowledged:
            raise TrainingSafetyError(
                "refusing to call distill(): pass --mode train and "
                f"{TRAINING_ACK_FLAG} (or set DISTILLERY_TRAINING_ACK="
                f"{TRAINING_ACK_PHRASE})"
            )
        return self._client.distill(dataset, recipe=recipe)


def _coerce_plan(raw: Any, requested: str) -> PlanResult:
    if isinstance(raw, PlanResult):
        return raw
    if isinstance(raw, Mapping):
        teacher = raw.get("teacher") or raw.get("models", {}).get("teacher", {})
        student = raw.get("student") or raw.get("models", {}).get("student", {})
        cost = raw.get("cost_estimate_usd") or raw.get("cost") or {}
        return PlanResult(
            requested_recipe=str(raw.get("requested_recipe", requested)),
            resolved_recipe=raw.get("resolved_recipe") or raw.get("resolved"),
            resolver_reasons=tuple(raw.get("resolver_reasons") or raw.get("reasons") or ()),
            do_not_distill=bool(raw.get("do_not_distill", False)),
            teacher_id=str(teacher.get("id", "Qwen/Qwen2.5-1.5B-Instruct")),
            student_id=str(student.get("id", "Qwen/Qwen2.5-0.5B-Instruct")),
            teacher_revision=str(teacher.get("revision", "UNPINNED_PENDING_PREFLIGHT")),
            student_revision=str(student.get("revision", "UNPINNED_PENDING_PREFLIGHT")),
            estimate_low_usd=float(cost.get("low", cost.get("estimate_low_usd", 0))),
            estimate_high_usd=float(cost.get("high", cost.get("estimate_high_usd", 0))),
            blockers=tuple(raw.get("blockers") or ()),
            details=dict(raw.get("details") or {}),
        )
    # Object with attributes
    return PlanResult(
        requested_recipe=str(getattr(raw, "requested_recipe", requested)),
        resolved_recipe=getattr(raw, "resolved_recipe", None),
        resolver_reasons=tuple(getattr(raw, "resolver_reasons", ()) or ()),
        do_not_distill=bool(getattr(raw, "do_not_distill", False)),
        teacher_id=str(getattr(raw, "teacher_id", "Qwen/Qwen2.5-1.5B-Instruct")),
        student_id=str(getattr(raw, "student_id", "Qwen/Qwen2.5-0.5B-Instruct")),
        teacher_revision=str(getattr(raw, "teacher_revision", "UNPINNED_PENDING_PREFLIGHT")),
        student_revision=str(getattr(raw, "student_revision", "UNPINNED_PENDING_PREFLIGHT")),
        estimate_low_usd=float(getattr(raw, "estimate_low_usd", 0)),
        estimate_high_usd=float(getattr(raw, "estimate_high_usd", 0)),
        blockers=tuple(getattr(raw, "blockers", ()) or ()),
        details=dict(getattr(raw, "details", {}) or {}),
    )


class LocalProtocolAdapter:
    """Fixture-backed adapter used when the SDK package is not installed.

    Stages: Curate → Synthesize (inventory) → Train-planned (plan only) → Prove
    (placeholder insufficient_evidence). Never submits jobs.
    """

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def create_dataset(self, path: Path) -> DatasetRef:
        raw = path.read_bytes()
        self._rows = load_dataset_jsonl(path)
        digest = sha256(raw).hexdigest()
        return DatasetRef(
            dataset_id=f"ds_local_{digest[:12]}",
            uri=path.resolve().as_uri(),
            content_sha256=digest,
            example_count=len(self._rows),
        )

    def plan_distillation(self, dataset: DatasetRef, *, recipe: str = "auto") -> PlanResult:
        label_sources: dict[str, int] = {}
        for row in self._rows:
            source = str(row.get("provenance", {}).get("label_source", "unknown"))
            label_sources[source] = label_sources.get(source, 0) + 1

        resolved: str | None = "sequence.v1"
        reasons = ("usable_responses_present", "local_adapter_plan_only")
        do_not_distill = False
        blockers: tuple[str, ...] = ()

        # Prefer contracts auto-resolver when available.
        try:
            from distillery.contracts.recipes import (  # type: ignore[import-not-found]
                AutoResolverInput,
                resolve_requested_recipe,
            )

            result = resolve_requested_recipe(
                recipe,
                auto_input=AutoResolverInput(usable_responses_exist=True),
            )
            resolved = result.resolved
            reasons = result.reasons
            do_not_distill = resolved == "do_not_distill"
            if result.error_code is not None:
                blockers = (result.error_code.value,)
        except ImportError:
            pass

        return PlanResult(
            requested_recipe=recipe,
            resolved_recipe=resolved,
            resolver_reasons=reasons,
            do_not_distill=do_not_distill,
            teacher_id="Qwen/Qwen2.5-1.5B-Instruct",
            student_id="Qwen/Qwen2.5-0.5B-Instruct",
            teacher_revision="UNPINNED_PENDING_PREFLIGHT",
            student_revision="UNPINNED_PENDING_PREFLIGHT",
            estimate_low_usd=0.0,
            estimate_high_usd=25.0,
            blockers=blockers,
            details={
                "backend": "local_protocol_adapter",
                "dataset_id": dataset.dataset_id,
                "label_source_counts": label_sources,
                "launches_training": False,
                "note": (
                    "Plan-only adapter. Model revisions remain unpinned until "
                    "Workstream 1 preflight lands. Real benchmarks pending."
                ),
            },
        )

    def distill(
        self,
        dataset: DatasetRef,
        *,
        recipe: str = "auto",
        training_acknowledged: bool = False,
    ) -> Any:
        del dataset, recipe
        if not training_acknowledged:
            raise TrainingSafetyError(
                "LocalProtocolAdapter refuses distill() without acknowledgment"
            )
        raise TrainingSafetyError(
            "LocalProtocolAdapter cannot launch training in this workstream "
            "(no model downloads, no SageMaker jobs). Use --mode plan or "
            "offline/precomputed fallback."
        )


def synthesize_inventory(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Synthesize stage: count label sources; do not call a teacher."""
    counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("provenance", {}).get("label_source", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    imported_or_oracle = sum(
        n for key, n in counts.items() if key in {"oracle", "imported", "teacher"}
    )
    return {
        "stage": "synthesize",
        "label_source_counts": counts,
        "usable_responses": imported_or_oracle,
        "teacher_calls_planned": 0,
        "skip_reason": "responses_already_present" if imported_or_oracle else None,
        "note": "Teacher is optional; oracle/imported labels skip generation.",
    }


def build_proof_placeholder(plan: PlanResult) -> ProofRef:
    status = "do_not_distill" if plan.do_not_distill else "insufficient_evidence"
    return ProofRef(
        report_id="prf_demo_pending",
        proof_status=status,
        limitations=(
            "Synthetic finance_world.v1 demo data only.",
            "Real arm metrics, paired CIs, and seed-23 replication are pending.",
            "Serving economics are projected until a later deployment study.",
            "This plan-only rehearsal did not train TinyFable.",
        ),
    )


def training_acknowledged(args: argparse.Namespace) -> bool:
    env_ack = os.environ.get("DISTILLERY_TRAINING_ACK") == TRAINING_ACK_PHRASE
    return bool(args.i_acknowledge_training_will_launch and env_ack) or bool(
        args.i_acknowledge_training_will_launch and args.ack_phrase == TRAINING_ACK_PHRASE
    )


def run_pipeline(
    *,
    dataset_path: Path,
    mode: str,
    recipe: str,
    seed: int,
    client: DistilleryClient,
    acknowledge_training: bool,
) -> dict[str, Any]:
    # Curate
    dataset = client.create_dataset(dataset_path)
    rows = load_dataset_jsonl(dataset_path)
    curate = {
        "stage": "curate",
        "dataset_id": dataset.dataset_id,
        "uri": dataset.uri,
        "content_sha256": dataset.content_sha256,
        "example_count": dataset.example_count,
    }

    # Synthesize (inventory only in plan/dry-run)
    synthesis = synthesize_inventory(rows)

    # Train-planned
    plan = client.plan_distillation(dataset, recipe=recipe)
    train_stage: dict[str, Any] = {
        "stage": "train",
        "mode": mode,
        "plan": plan.as_dict(),
        "training_launched": False,
    }

    if mode == "train":
        if not acknowledge_training:
            raise TrainingSafetyError(
                "training mode requested without safety acknowledgment; "
                f"re-run with {TRAINING_ACK_FLAG} and "
                f"--ack-phrase {TRAINING_ACK_PHRASE}"
            )
        run = client.distill(
            dataset, recipe=recipe, training_acknowledged=True
        )
        train_stage["training_launched"] = True
        train_stage["run"] = repr(run)
    elif mode in {"plan", "dry-run", "offline"}:
        train_stage["training_launched"] = False
    else:
        raise ValueError(f"unknown mode: {mode}")

    # Held-out selector (public inputs only)
    try:
        selection = select_held_out(rows, seed=seed, per_task=1)
        held_out = {
            "seed": selection.seed,
            "example_ids": list(selection.example_ids),
            "examples_public": selection.as_public_view(),
        }
    except ValueError as exc:
        held_out = {"error": str(exc), "seed": seed}

    proof = build_proof_placeholder(plan)
    prove = {
        "stage": "prove",
        "report_id": proof.report_id,
        "proof_status": proof.proof_status,
        "limitations": list(proof.limitations),
        "held_out": held_out,
    }

    return {
        "product": "Distillery",
        "tagline": "Smaller models. Proven economics.",
        "model_name": "TinyFable",
        "generated_at": datetime.now(UTC).isoformat(),
        "stages": {
            "curate": curate,
            "synthesize": synthesis,
            "train": train_stage,
            "prove": prove,
        },
        "claims": {
            "honest": True,
            "benchmarks": "pending",
            "training_launched": train_stage["training_launched"],
            "data": "synthetic_finance_world_v1",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TinyFable finance-generalist Distillery demo (plan-only by default)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to finance_world JSONL",
    )
    parser.add_argument(
        "--mode",
        choices=("plan", "dry-run", "offline", "train"),
        default="plan",
        help="plan/dry-run/offline never train; train requires acknowledgment",
    )
    parser.add_argument("--recipe", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_DEMO_SEED)
    parser.add_argument(
        "--api-key",
        default=None,
        help="DISTILLERY_API_KEY; if unset and SDK missing, uses local adapter",
    )
    parser.add_argument(
        TRAINING_ACK_FLAG,
        dest="i_acknowledge_training_will_launch",
        action="store_true",
        help="Required safety gate for --mode train",
    )
    parser.add_argument(
        "--ack-phrase",
        default="",
        help=f"Must equal {TRAINING_ACK_PHRASE} together with the ack flag for train mode",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.dataset.is_file():
        print(f"error: dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    client: DistilleryClient = try_import_sdk_client(args.api_key) or LocalProtocolAdapter()
    ack = training_acknowledged(args)

    try:
        result = run_pipeline(
            dataset_path=args.dataset,
            mode=args.mode,
            recipe=args.recipe,
            seed=args.seed,
            client=client,
            acknowledge_training=ack,
        )
    except TrainingSafetyError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        stages = result["stages"]
        print("Distillery · TinyFable finance generalist")
        print("Smaller models. Proven economics.")
        print()
        print(f"[Curate]     dataset={stages['curate']['dataset_id']} "
              f"n={stages['curate']['example_count']} "
              f"sha256={stages['curate']['content_sha256'][:12]}…")
        synth = stages["synthesize"]
        print(
            f"[Synthesize] usable={synth['usable_responses']} "
            f"teacher_calls_planned={synth['teacher_calls_planned']} "
            f"skip={synth['skip_reason']}"
        )
        plan = stages["train"]["plan"]
        print(
            f"[Train]      mode={stages['train']['mode']} "
            f"resolved={plan['resolved_recipe']} "
            f"launched={stages['train']['training_launched']}"
        )
        print(
            f"[Prove]      status={stages['prove']['proof_status']} "
            f"(real benchmarks pending)"
        )
        print()
        print("Limitations:")
        for item in stages["prove"]["limitations"]:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
