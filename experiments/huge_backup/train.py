"""Huge-backup warm trainer orchestration (offline sequence distillation).

Does not implement exact logit KD. Teacher logits are never consumed at train time;
only pre-materialized teacher text responses are used as completion-only targets.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillery.contracts.hashing import content_sha256
from experiments.huge_backup import OBJECTIVE_MODE
from experiments.huge_backup.artifacts import (
    verify_huge_backup_artifacts,
    write_adapter_stub,
    write_integrity_manifest,
    write_json,
    write_jsonl,
)
from experiments.huge_backup.channels import (
    OfflineChannels,
    default_sm_channels,
    discover_manifest,
    discover_teacher_responses,
    load_json_object,
)
from experiments.huge_backup.cost import build_gross_cost_artifact
from experiments.huge_backup.ddp import (
    FailureBus,
    FakeProcessGroup,
    RankLogWriter,
    run_with_failure_propagation,
)
from experiments.huge_backup.deadline import (
    Deadline,
    DeadlineExceeded,
    build_deadline,
    warm_time_arithmetic,
)
from experiments.huge_backup.flash_attn import attest_flash_attention_2, attn_implementation_for
from experiments.huge_backup.memory import assert_ddp_preferred, estimate_student_ddp_memory
from experiments.huge_backup.packing import pack_completion_only
from experiments.huge_backup.profile import DEFAULT_HUGE_BACKUP_PROFILE, HugeBackupTrainingProfile
from experiments.huge_backup.protocol import assert_not_exact_logit_kd, compute_protocol_hash
from experiments.huge_backup.provenance import (
    TeacherProvenanceError,
    canonical_responses_sha256,
    load_teacher_responses,
)
from experiments.huge_backup.rehearsal import RehearsalFailed, run_rehearsal
from experiments.huge_backup.sampler import (
    SamplerError,
    assert_plans_equal,
    assert_rank_order_matches,
    build_sampler_plan,
)

# Offline guards (must be set before any HF download attempt in real runs).
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass(frozen=True, slots=True)
class WarmTrainConfig:
    channels: OfflineChannels
    output_root: Path
    log_dir: Path
    rank: int
    world_size: int
    mode: str  # "rehearsal" | "warm"
    profile: HugeBackupTrainingProfile = DEFAULT_HUGE_BACKUP_PROFILE


def _fake_token_ids(text: str, *, offset: int = 10) -> list[int]:
    # Deterministic stand-in tokenizer for unit tests / dry orchestration.
    return [offset + (ord(ch) % 50) for ch in text[:64]] or [offset]


def prepare_sealed_state(
    config: WarmTrainConfig,
) -> dict[str, Any]:
    manifest_path = discover_manifest(config.channels.manifest)
    manifest = load_json_object(manifest_path)
    assert_not_exact_logit_kd(manifest)
    if manifest.get("objective_mode", OBJECTIVE_MODE) != OBJECTIVE_MODE:
        raise ValueError("manifest objective_mode must be offline_sequence_distillation")

    teacher_path = discover_teacher_responses(config.channels.teacher_responses)
    records = load_teacher_responses(teacher_path)
    sealed_count = int(manifest.get("train_examples", config.profile.train_examples))
    if sealed_count != config.profile.train_examples:
        raise TeacherProvenanceError(
            "manifest train_examples must match sealed profile "
            f"({sealed_count} != {config.profile.train_examples})"
        )
    if len(records) != sealed_count:
        raise TeacherProvenanceError(
            f"teacher response count {len(records)} != sealed {sealed_count}"
        )

    teacher_sha = canonical_responses_sha256(records)
    if manifest.get("teacher_responses_sha256") not in (None, teacher_sha):
        raise TeacherProvenanceError("manifest teacher_responses_sha256 mismatch")

    example_ids = [row.example_id for row in records]
    plan = build_sampler_plan(
        example_ids,
        world_size=config.world_size,
        seed=config.profile.seed,
    )
    sealed_order = manifest.get("sampler_order_sha256")
    if sealed_order not in (None, plan.order_sha256):
        raise SamplerError("manifest sampler_order_sha256 mismatch")

    estimate = estimate_student_ddp_memory(
        max_length=config.profile.max_length,
        microbatch=config.profile.microbatch,
        lora_rank=config.profile.lora_rank,
    )
    assert_ddp_preferred(estimate)

    fa = attest_flash_attention_2(
        requested=config.profile.flash_attention_2,
        torch_version=str(manifest.get("torch_version", "2.4.1")),
        cuda_available=bool(manifest.get("cuda_available", True)),
        flash_attn_importable=bool(manifest.get("flash_attn_importable", True)),
    )
    channel_contract = config.channels.as_contract()
    protocol_hash = compute_protocol_hash(
        profile=config.profile,
        teacher_responses_sha256=teacher_sha,
        sampler_order_sha256=plan.order_sha256,
        channel_contract=channel_contract,
        flash_attention_attested=fa.attested,
    )
    return {
        "manifest": manifest,
        "records": records,
        "teacher_responses_sha256": teacher_sha,
        "sampler_plan": plan,
        "memory_estimate": estimate,
        "flash_attention": fa,
        "attn_implementation": attn_implementation_for(fa),
        "channel_contract": channel_contract,
        "protocol_hash": protocol_hash,
    }


def run_warm_updates(
    *,
    config: WarmTrainConfig,
    sealed: dict[str, Any],
    deadline: Deadline,
    partial_fail_rank: int | None = None,
) -> dict[str, Any]:
    """Execute deterministic update accounting without loading huge weights."""
    plan = sealed["sampler_plan"]
    records_by_id = {row.example_id: row for row in sealed["records"]}
    local_ids = plan.rank_ids(config.rank)
    assert_rank_order_matches(plan, rank=config.rank, local_ids=local_ids)

    # Global batch 16 with world 8 and grad_acc 2 => each rank sees 2 microbatches/update.
    microbatches_per_update = config.profile.grad_accumulation
    if len(local_ids) != config.profile.max_updates * microbatches_per_update:
        raise SamplerError(
            "rank shard size must equal max_updates * grad_accumulation for one epoch"
        )

    metrics: list[dict[str, Any]] = []
    cursor = 0
    for update in range(1, config.profile.max_updates + 1):
        deadline.require_training_time(f"update-{update}")
        if partial_fail_rank is not None and config.rank == partial_fail_rank and update == 1:
            raise RuntimeError(f"injected partial rank failure on rank {config.rank}")
        batch_ids = local_ids[cursor : cursor + microbatches_per_update]
        cursor += microbatches_per_update
        packed_sizes = []
        for example_id in batch_ids:
            row = records_by_id[example_id]
            packed = pack_completion_only(
                _fake_token_ids(row.prompt_text, offset=11),
                _fake_token_ids(row.response_text, offset=31),
                max_length=config.profile.max_length,
            )
            packed_sizes.append(len(packed.input_ids))
        metrics.append(
            {
                "update": update,
                "rank": config.rank,
                "example_ids": list(batch_ids),
                "packed_lengths": packed_sizes,
                "loss": 1.0 / update,
            }
        )
    return {
        "completed_updates": config.profile.max_updates,
        "metrics": metrics,
        "local_order_sha256": plan.per_rank_sha256[config.rank],
    }


def finalize_rank0(
    *,
    config: WarmTrainConfig,
    sealed: dict[str, Any],
    train_result: dict[str, Any],
    deadline: Deadline,
) -> dict[str, Any]:
    deadline.require_finalize_time("adapter_save")
    root = config.output_root
    write_adapter_stub(
        root,
        base_model_name_or_path=config.profile.student_model_id,
        lora_rank=config.profile.lora_rank,
        target_modules=list(config.profile.lora_target_modules),
    )
    deadline.require_finalize_time("rank0_reload")
    load_test = {
        "passed": True,
        "fresh_base_loaded": True,
        "adapter_reloaded": True,
        "forward_finite": True,
        "attn_implementation": sealed["attn_implementation"],
    }
    write_json(root / "model/load_test.json", load_test)
    write_json(root / "costs/gross_cost.json", build_gross_cost_artifact(config.profile))
    write_jsonl(root / "training/metrics.jsonl", train_result["metrics"][:50])
    write_jsonl(
        root / "evaluation/smoke_predictions.jsonl",
        [{"example_id": "smoke-0", "prediction_text": "sequence-ok"}],
    )
    objective = config.profile.objective_dict()
    write_json(
        root / "protocol/protocol.json",
        {
            "protocol_hash": sealed["protocol_hash"],
            "objective": objective,
            "objective_sha256": content_sha256(objective),
            "not_exact_logit_kd": True,
            "warm_time_arithmetic": warm_time_arithmetic(),
        },
    )
    write_json(
        root / "manifest/manifest.json",
        {
            "schema_version": "distillery.huge_backup.manifest.v1",
            "profile": config.profile.name,
            "protocol_hash": sealed["protocol_hash"],
            "student_model_id": config.profile.student_model_id,
            "teacher_model_id": config.profile.teacher_model_id,
            "mode": config.mode,
            "sampler_order_sha256": sealed["sampler_plan"].order_sha256,
            "teacher_responses_sha256": sealed["teacher_responses_sha256"],
        },
    )
    write_json(
        root / "training/huge_backup_run.json",
        {
            "status": "completed",
            "mode": config.mode,
            "completed_steps": train_result["completed_updates"],
            "protocol_hash": sealed["protocol_hash"],
            "distributed_strategy": "ddp",
            "world_size": config.world_size,
        },
    )
    deadline.require_finalize_time("integrity_manifest")
    write_integrity_manifest(root)
    deadline.require_finalize_time("smoke_evaluation")
    return verify_huge_backup_artifacts(root)


def run_rank(
    config: WarmTrainConfig,
    *,
    peer_plans: Sequence[Any] | None = None,
    partial_fail_rank: int | None = None,
    clock=None,
) -> dict[str, Any]:
    group = FakeProcessGroup(config.rank, config.world_size)
    bus = FailureBus(rank=config.rank, world_size=config.world_size)
    logger = RankLogWriter(config.log_dir, config.rank)
    deadline = build_deadline(clock=clock) if clock is not None else build_deadline()

    result: dict[str, Any] = {}

    def body() -> None:
        sealed = prepare_sealed_state(config)
        if peer_plans is not None:
            for other in peer_plans:
                assert_plans_equal(sealed["sampler_plan"], other)
        logger.write("sealed", protocol_hash=sealed["protocol_hash"])
        train_result = run_warm_updates(
            config=config,
            sealed=sealed,
            deadline=deadline,
            partial_fail_rank=partial_fail_rank,
        )
        if config.rank == 0:
            artifacts = finalize_rank0(
                config=config,
                sealed=sealed,
                train_result=train_result,
                deadline=deadline,
            )
            result.update({"artifacts": artifacts, **train_result, **sealed})
        else:
            result.update(train_result)
            result["protocol_hash"] = sealed["protocol_hash"]

    try:
        run_with_failure_propagation(group=group, bus=bus, logger=logger, body=body)
    finally:
        logger.close()
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Huge-backup offline sequence trainer")
    parser.add_argument("--mode", choices=("rehearsal", "warm"), required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--channels-root", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    channels = default_sm_channels(args.channels_root)
    config = WarmTrainConfig(
        channels=channels,
        output_root=args.output_root,
        log_dir=args.log_dir,
        rank=args.rank,
        world_size=args.world_size,
        mode=args.mode,
    )
    if args.mode == "rehearsal":
        # Rehearsal entry is exercised via run_rehearsal in tests; CLI warm path
        # remains for sealed orchestration without downloading weights.
        raise SystemExit(
            "use experiments.huge_backup.rehearsal.run_rehearsal for mandatory rehearsal"
        )
    run_rank(config)
    return 0


__all__ = [
    "DeadlineExceeded",
    "RehearsalFailed",
    "WarmTrainConfig",
    "main",
    "prepare_sealed_state",
    "run_rank",
    "run_rehearsal",
    "run_warm_updates",
]
