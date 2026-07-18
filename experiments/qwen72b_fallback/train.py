"""Real eight-rank 72B QLoRA probe/rehearsal/full trainer."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from experiments.qwen72b_fallback.artifacts import (
    artifact_bundle_sha256,
    seal_existing_run_artifacts,
)
from experiments.qwen72b_fallback.ddp import (
    DistributedContext,
    RankLogWriter,
    acknowledge_all_ranks,
    initialize_distributed,
    require_equal_shapes,
    run_synchronized_phase,
    shutdown_distributed,
)
from experiments.qwen72b_fallback.deadline import (
    RunPhase,
    build_deadline,
)
from experiments.qwen72b_fallback.evidence import sha256_file
from experiments.qwen72b_fallback.finance_world_targets import (
    FinanceWorldCorpusEvidence,
    FinanceWorldTargetRecord,
)
from experiments.qwen72b_fallback.license_policy import (
    ATTRIBUTION_PLAN_PATH,
    QWEN_NOTICE_PATH,
)
from experiments.qwen72b_fallback.memory import (
    Qwen72BMemoryProbeEvidence,
    memory_probe_measurement_sha256,
    require_measured_probe,
)
from experiments.qwen72b_fallback.packing import (
    collate_fixed_shape,
    pack_completion_only,
)
from experiments.qwen72b_fallback.pins import (
    MODEL_ID,
    REVISION,
    load_weight_inventory,
)
from experiments.qwen72b_fallback.profile import (
    Qwen72BTrainingProfile,
    RunKind,
)
from experiments.qwen72b_fallback.protocol import build_protocol
from experiments.qwen72b_fallback.readiness import (
    ExecutionAction,
    ExecutionAuthorization,
)
from experiments.qwen72b_fallback.sampler import build_sampler_plan

T = TypeVar("T")
RANK0_PHASE_POLL_SECONDS = 1.0
RANK0_PHASE_TIMEOUT_SECONDS = 1200


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rank0_filesystem_phase(
    context: DistributedContext,
    *,
    status_path: Path,
    phase: str,
    operation: Callable[[], T],
    timeout_seconds: int = RANK0_PHASE_TIMEOUT_SECONDS,
) -> T | None:
    """Allow a long rank-0-only operation without holding an NCCL collective."""

    def clear_stale_status() -> bool:
        if context.is_writer:
            status_path.unlink(missing_ok=True)
        return True

    run_synchronized_phase(context, f"{phase}_prepare", clear_stale_status)
    if context.is_writer:
        try:
            result = operation()
            atomic_json(status_path, {"phase": phase, "ok": True})
        except BaseException as exc:  # noqa: BLE001 - propagate to every rank
            atomic_json(
                status_path,
                {
                    "phase": phase,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            result = None
    else:
        result = None
    deadline = time.monotonic() + timeout_seconds
    while not status_path.is_file() and time.monotonic() < deadline:
        time.sleep(RANK0_PHASE_POLL_SECONDS)
    if not status_path.is_file():
        raise RuntimeError(f"rank-0 phase status timed out: {phase}")
    status = json.loads(status_path.read_bytes())

    def acknowledge() -> dict[str, Any]:
        if status.get("phase") != phase or status.get("ok") is not True:
            raise RuntimeError(
                f"rank-0 phase {phase} failed: {status.get('error', 'invalid status')}"
            )
        return status

    run_synchronized_phase(context, f"{phase}_ack", acknowledge)
    return result


def resolve_snapshot(models_dir: Path) -> Path:
    candidates = (
        models_dir,
        models_dir / "Qwen" / "Qwen2.5-72B-Instruct" / REVISION,
    )
    matches = [path for path in candidates if (path / "config.json").is_file()]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one local 72B snapshot, found {matches}")
    return matches[0]


def verify_snapshot_files(snapshot: Path) -> dict[str, str]:
    inventory = load_weight_inventory()
    actual_names = {
        path.relative_to(snapshot).as_posix() for path in snapshot.iterdir() if path.is_file()
    }
    if not set(inventory.files) <= actual_names:
        raise ValueError(
            f"local snapshot is incomplete: {sorted(set(inventory.files) - actual_names)}"
        )
    hashes: dict[str, str] = {}
    for name, expected in sorted(inventory.files.items()):
        path = snapshot / name
        if path.stat().st_size != expected.size:
            raise ValueError(f"local snapshot size mismatch: {name}")
        digest = sha256_file(path)
        if digest != expected.sha256:
            raise ValueError(f"local snapshot body hash mismatch: {name}")
        hashes[name] = digest
    return hashes


def verify_image_runtime(
    authorization: ExecutionAuthorization,
    *,
    runtime_image_digest: str,
    version_path: Path = Path("/opt/distillery/VERSION.json"),
) -> None:
    image = authorization.evidence_bundle.ecr_image
    if image is None:
        raise ValueError("training authorization lacks exact ECR image evidence")
    if runtime_image_digest != image.image_digest:
        raise ValueError("runtime image digest differs from authorization")
    version = json.loads(version_path.read_bytes())
    expected = {
        "source_sha": image.source_revision,
        "source_tree_sha256": image.source_tree_sha256,
        "package_lock_sha256": image.package_lock_sha256,
        "qwen72b_trainer_module": "experiments.qwen72b_fallback.train",
        "qwen72b_attention_backend": "sdpa_math",
        "flash_attention_2_packaged": False,
    }
    for key, value in expected.items():
        if version.get(key) != value:
            raise ValueError(f"runtime image VERSION.json mismatch: {key}")


def load_finance_data(
    data_dir: Path,
    authorization: ExecutionAuthorization,
    profile: Qwen72BTrainingProfile,
) -> tuple[FinanceWorldCorpusEvidence, dict[str, FinanceWorldTargetRecord]]:
    manifest_path = data_dir / "finance_world_evidence.json"
    records_path = data_dir / "train.jsonl"
    evidence = FinanceWorldCorpusEvidence.model_validate_json(manifest_path.read_bytes())
    authorized = authorization.evidence_bundle.finance_world_data
    if authorized is None or evidence != authorized:
        raise ValueError("finance-world channel evidence differs from authorization")
    records: dict[str, FinanceWorldTargetRecord] = {}
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = FinanceWorldTargetRecord.model_validate_json(line)
                records[record.example_id] = record
    expected_records = {record.example_id: record for record in evidence.records}
    if records != expected_records:
        raise ValueError("finance-world JSONL differs from its hash-bound evidence")
    if len(records) != profile.train_examples:
        raise ValueError("finance-world channel record count differs from profile")
    return evidence, records


def set_determinism(torch: Any, seed: int) -> dict[str, Any]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return {
        "torch_deterministic_algorithms": True,
        "sdpa_flash_enabled": False,
        "sdpa_mem_efficient_enabled": False,
        "sdpa_math_enabled": True,
        "bitwise_reproducibility_claimed": False,
        "known_limit": (
            "bitsandbytes NF4 kernels are measured with deterministic-algorithm "
            "enforcement but are not claimed bitwise reproducible across driver stacks"
        ),
    }


def load_tokenizer(snapshot: Path) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Qwen tokenizer lacks pad and eos tokens")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_qlora_model(
    snapshot: Path,
    profile: Qwen72BTrainingProfile,
) -> Any:
    import torch
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=profile.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora = LoraConfig(
        r=profile.lora_rank,
        lora_alpha=profile.lora_alpha,
        lora_dropout=profile.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(profile.lora_target_modules),
    )
    model = get_peft_model(model, lora)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("PEFT model exposes no trainable adapter parameters")
    return model


def tokenized_batch(
    tokenizer: Any,
    record: FinanceWorldTargetRecord,
    *,
    max_length: int,
) -> dict[str, list[list[int]] | list[list[float]]]:
    messages = [{"role": "user", "content": record.prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": record.target_text}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Qwen chat template completion prefix is not stable")
    completion_ids = full_ids[len(prompt_ids) :]
    packed = pack_completion_only(
        list(prompt_ids),
        list(completion_ids),
        max_length=max_length,
    )
    return collate_fixed_shape(
        [packed],
        max_length=max_length,
        pad_token_id=int(tokenizer.pad_token_id),
    )


def to_tensors(torch: Any, batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long, device="cuda:0"),
        "attention_mask": torch.tensor(
            batch["attention_mask"],
            dtype=torch.long,
            device="cuda:0",
        ),
        "labels": torch.tensor(batch["labels"], dtype=torch.long, device="cuda:0"),
    }


def execute_optimizer_step(
    torch: Any,
    optimizer: Any,
    wrapped: Any,
    tensors: dict[str, Any],
) -> float:
    optimizer.zero_grad(set_to_none=True)
    output = wrapped(**tensors)
    loss = output.loss
    if loss is None or not bool(torch.isfinite(loss).item()):
        raise RuntimeError("training loss is absent or non-finite")
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def save_peft_adapter(wrapped: Any, adapter_dir: Path) -> dict[str, str]:
    wrapped.module.save_pretrained(
        adapter_dir,
        safe_serialization=True,
    )
    return adapter_checksums(adapter_dir)


def adapter_checksums(adapter_dir: Path) -> dict[str, str]:
    config = adapter_dir / "adapter_config.json"
    weights = adapter_dir / "adapter_model.safetensors"
    if not config.is_file() or not weights.is_file() or weights.stat().st_size == 0:
        raise RuntimeError("PEFT adapter save lacks real config/weights")
    from safetensors import safe_open

    with safe_open(weights, framework="pt", device="cpu") as handle:
        names = list(handle.keys())
        if not names:
            raise RuntimeError("adapter safetensors contains no tensors")
        if any(handle.get_tensor(name).numel() == 0 for name in names):
            raise RuntimeError("adapter safetensors contains an empty tensor")
    return {
        path.relative_to(adapter_dir.parent.parent).as_posix(): sha256_file(path)
        for path in sorted(adapter_dir.iterdir())
        if path.is_file()
    }


def reload_adapter_forward_probe(
    *,
    snapshot: Path,
    adapter_dir: Path,
    batch: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    checksums_before = adapter_checksums(adapter_dir)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    reloaded = PeftModel.from_pretrained(
        base,
        str(adapter_dir),
        is_trainable=False,
        local_files_only=True,
    )
    tensors = to_tensors(torch, batch)
    with torch.inference_mode():
        output = reloaded(
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
        )
    finite = bool(torch.isfinite(output.logits).all().item())
    checksums_after = adapter_checksums(adapter_dir)
    if checksums_before != checksums_after:
        raise RuntimeError("adapter files changed during PEFT reload")
    if not finite:
        raise RuntimeError("reloaded PEFT adapter forward probe produced non-finite logits")
    return {
        "fresh_base_loaded": True,
        "peft_adapter_reloaded": True,
        "forward_finite": True,
        "adapter_checksums": checksums_after,
        "logits_shape": list(output.logits.shape),
    }


def _action_for_mode(mode: str) -> ExecutionAction:
    return {
        "memory_probe": ExecutionAction.MEMORY_PROBE,
        "rehearsal": ExecutionAction.REHEARSAL,
        "full": ExecutionAction.FULL,
    }[mode]


def run_child(args: argparse.Namespace) -> int:
    import torch
    from torch.nn.parallel import DistributedDataParallel

    context = initialize_distributed(timeout_seconds=120, torch_module=torch)
    logger = RankLogWriter(args.output_dir / "rank-logs", context.rank)
    failure_path = args.output_dir / "failures" / f"rank-{context.rank:02d}.json"
    profile = Qwen72BTrainingProfile.model_validate_json(args.profile.read_bytes())
    authorization = ExecutionAuthorization.model_validate_json(args.authorization.read_bytes())
    action = _action_for_mode(args.mode)
    deadline = build_deadline("memory_probe" if args.mode == "memory_probe" else profile.kind.value)
    try:
        authorization.require_current(action=action, launch_name=args.launch_name)
        if args.mode != profile.kind.value and not (
            args.mode == "memory_probe" and profile.kind in {RunKind.REHEARSAL, RunKind.FULL}
        ):
            raise ValueError("child mode differs from target profile")
        runtime_digest = args.runtime_image_digest
        verify_image_runtime(
            authorization,
            runtime_image_digest=runtime_digest,
            version_path=args.version_path,
        )
        protocol = build_protocol(
            profile=profile,
            authorization=authorization,
        )
        if context.is_writer:
            atomic_json(
                args.output_dir / "protocol.json",
                protocol.model_dump(mode="json"),
            )
            atomic_json(
                args.output_dir / "profile.json",
                profile.model_dump(mode="json"),
            )
            (args.output_dir / "compliance").mkdir(parents=True, exist_ok=True)
            (args.output_dir / "compliance" / "QWEN_NOTICE.txt").write_bytes(
                QWEN_NOTICE_PATH.read_bytes()
            )
            (args.output_dir / "compliance" / "attribution_plan.json").write_bytes(
                ATTRIBUTION_PLAN_PATH.read_bytes()
            )
        set_determinism(torch, profile.seed)
        snapshot = resolve_snapshot(args.models_dir)
        channel_status = args.output_dir / "coordination" / "channel-verify.json"

        def verify_channels() -> None:
            deadline.require_phase_time(RunPhase.CHANNEL_VERIFY, "channel verification")
            hashes = verify_snapshot_files(snapshot)
            evidence, _records = load_finance_data(
                args.data_dir,
                authorization,
                profile,
            )
            atomic_json(
                args.output_dir / "channel-evidence.json",
                {
                    "snapshot_object_sha256": hashes,
                    "finance_world_evidence_sha256": evidence.evidence_sha256,
                },
            )

        rank0_filesystem_phase(
            context,
            status_path=channel_status,
            phase="channel_verify",
            operation=verify_channels,
        )
        finance_evidence, records = load_finance_data(
            args.data_dir,
            authorization,
            profile,
        )
        tokenizer = load_tokenizer(snapshot)
        sampler = build_sampler_plan(
            list(records),
            world_size=context.world_size,
            seed=profile.seed,
            expected_count=profile.train_examples,
        )
        rank_ids = sampler.rank_ids(context.rank)
        if len(rank_ids) != profile.max_updates:
            raise ValueError("deterministic rank shard differs from profile update count")
        first_batch = tokenized_batch(
            tokenizer,
            records[rank_ids[0]],
            max_length=profile.max_length,
        )
        require_equal_shapes(
            context,
            (
                len(first_batch["input_ids"]),
                len(first_batch["input_ids"][0]),
            ),
        )

        deadline.require_phase_time(RunPhase.MODEL_LOAD, "model load")
        torch.cuda.reset_peak_memory_stats(0)
        model = run_synchronized_phase(
            context,
            "model_load",
            lambda: load_qlora_model(snapshot, profile),
        )
        wrapped = DistributedDataParallel(
            model,
            device_ids=[0],
            output_device=0,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in wrapped.parameters() if parameter.requires_grad),
            lr=profile.learning_rate,
        )
        metrics_path = args.output_dir / "metrics" / f"rank-{context.rank:02d}.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        steps_to_run = 1 if args.mode == "memory_probe" else profile.max_updates
        with metrics_path.open("w", encoding="utf-8") as metrics:
            for step, example_id in enumerate(rank_ids[:steps_to_run], start=1):
                deadline.require_phase_time(RunPhase.TRAIN_STEPS, f"optimizer step {step}")
                batch = tokenized_batch(
                    tokenizer,
                    records[example_id],
                    max_length=profile.max_length,
                )
                shape = (len(batch["input_ids"]), len(batch["input_ids"][0]))
                require_equal_shapes(context, shape)
                tensors = to_tensors(torch, batch)

                loss_value = run_synchronized_phase(
                    context,
                    f"optimizer_step_{step}",
                    lambda tensors=tensors: execute_optimizer_step(
                        torch,
                        optimizer,
                        wrapped,
                        tensors,
                    ),
                )
                metrics.write(
                    json.dumps(
                        {
                            "rank": context.rank,
                            "step": step,
                            "example_id": example_id,
                            "loss": loss_value,
                            "shape": list(shape),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                metrics.flush()
                os.fsync(metrics.fileno())

        if args.mode == "memory_probe":
            peak = int(torch.cuda.max_memory_reserved(0))
            capacity = int(torch.cuda.get_device_properties(0).total_memory)
            measurement = {
                "rank": context.rank,
                "peak": peak,
                "capacity": capacity,
                "device": str(torch.cuda.get_device_name(0)),
                "shape": [1, profile.max_length],
                "peak_metric": "torch.cuda.max_memory_reserved",
            }
            gathered: list[dict[str, Any] | None] = [None] * context.world_size
            context.dist.all_gather_object(gathered, measurement)
            acknowledgements = acknowledge_all_ranks(context)
            if context.is_writer:
                rows = [row for row in gathered if isinstance(row, dict)]
                if len(rows) != context.world_size:
                    raise RuntimeError("memory probe did not collect every rank")
                device_names = tuple(row["device"] for row in rows)
                peaks = tuple(row["peak"] for row in rows)
                capacities = tuple(row["capacity"] for row in rows)
                shapes = tuple(tuple(row["shape"]) for row in rows)
                image = authorization.evidence_bundle.ecr_image
                if image is None:
                    raise ValueError("probe authorization lacks image evidence")
                probe = Qwen72BMemoryProbeEvidence.seal(
                    probe_id=args.launch_name,
                    revision=REVISION,
                    model_identity_sha256=(
                        authorization.evidence_bundle.local_policy.identity.evidence_sha256
                    ),
                    runtime_image_uri=image.image_uri,
                    runtime_image_digest=image.image_digest,
                    image_binding_sha256=image.image_binding_sha256,
                    profile_sha256=profile.profile_sha256,
                    device_names=device_names,
                    per_rank_peak_memory_bytes=peaks,
                    per_rank_capacity_bytes=capacities,
                    measured_batch_shapes=shapes,
                    model_load_completed=True,
                    forward_completed=True,
                    backward_completed=True,
                    optimizer_step_completed=True,
                    all_rank_acknowledgements=acknowledgements,
                    sampler_order_sha256=sampler.order_sha256,
                    probe_artifact_sha256=memory_probe_measurement_sha256(
                        profile_sha256=profile.profile_sha256,
                        device_names=device_names,
                        peaks=peaks,
                        capacities=capacities,
                        shapes=shapes,
                        sampler_order_sha256=sampler.order_sha256,
                    ),
                )
                atomic_json(
                    args.output_dir / "memory-probe.json",
                    probe.model_dump(mode="json"),
                )
            if context.is_writer:
                atomic_json(
                    args.output_dir / "completion" / "all-ranks.json",
                    {
                        "acknowledged_ranks": list(range(context.world_size)),
                        "acknowledgements": list(acknowledgements),
                        "mode": args.mode,
                    },
                )
            logger.write("rank_complete", mode=args.mode)
            return 0

        if args.memory_probe is None:
            raise ValueError("rehearsal/full execution requires a measured probe file")
        probe = Qwen72BMemoryProbeEvidence.model_validate_json(args.memory_probe.read_bytes())
        authorized_probe = authorization.evidence_bundle.memory_probe
        image = authorization.evidence_bundle.ecr_image
        if image is None or authorized_probe is None or probe != authorized_probe:
            raise ValueError("memory probe file differs from authorization evidence")
        require_measured_probe(
            probe,
            profile_sha256=profile.profile_sha256,
            model_identity_sha256=(
                authorization.evidence_bundle.local_policy.identity.evidence_sha256
            ),
            image_binding_sha256=image.image_binding_sha256,
            runtime_image_digest=image.image_digest,
        )

        deadline.require_phase_time(RunPhase.ADAPTER_SAVE, "adapter save")
        adapter_dir = args.output_dir / "model" / "adapter"

        rank0_filesystem_phase(
            context,
            status_path=args.output_dir / "coordination" / "adapter-save.json",
            phase="adapter_save",
            operation=lambda: save_peft_adapter(wrapped, adapter_dir),
        )
        optimizer = None
        wrapped = None
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        run_synchronized_phase(context, "model_release", lambda: True)

        deadline.require_phase_time(RunPhase.ADAPTER_RELOAD, "adapter reload")
        reload_status_path = args.output_dir / "coordination" / "adapter-reload.json"

        def reload_probe() -> dict[str, Any]:
            report = reload_adapter_forward_probe(
                snapshot=snapshot,
                adapter_dir=adapter_dir,
                batch=first_batch,
            )
            atomic_json(args.output_dir / "reload-probe.json", report)
            return report

        rank0_filesystem_phase(
            context,
            status_path=reload_status_path,
            phase="adapter_reload",
            operation=reload_probe,
        )
        deadline.require_phase_time(RunPhase.CLEANUP_ARTIFACTS, "artifact sealing")
        if context.is_writer:
            atomic_json(
                args.output_dir / "finance-world-evidence.json",
                finance_evidence.model_dump(mode="json"),
            )
            atomic_json(
                args.output_dir / "run-evidence.json",
                {
                    "authorization_sha256": authorization.evidence_sha256,
                    "profile_sha256": profile.profile_sha256,
                    "memory_probe_sha256": probe.evidence_sha256,
                    "sampler_order_sha256": sampler.order_sha256,
                    "runtime_image_digest": runtime_digest,
                    "model_id": MODEL_ID,
                    "revision": REVISION,
                    "flash_attention_2": False,
                    "attention_backend": "sdpa_math",
                },
            )
            entries = seal_existing_run_artifacts(args.output_dir)
            atomic_json(
                args.output_dir / "bundle.json",
                {
                    "artifact_sha256": entries,
                    "bundle_sha256": artifact_bundle_sha256(entries),
                },
            )
        acknowledgements = acknowledge_all_ranks(context)
        if context.is_writer:
            atomic_json(
                args.output_dir / "completion" / "all-ranks.json",
                {
                    "acknowledged_ranks": list(range(context.world_size)),
                    "acknowledgements": list(acknowledgements),
                    "mode": args.mode,
                    "real_peft_adapter_saved": True,
                    "real_peft_reload_forward_probe": True,
                },
            )
        logger.write("rank_complete", mode=args.mode)
        return 0
    except BaseException as exc:  # noqa: BLE001 - preserve complete rank failure
        atomic_json(
            failure_path,
            {
                "rank": context.rank,
                "mode": args.mode,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        logger.write("rank_failure", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        logger.close()
        shutdown_distributed(context)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("memory_probe", "rehearsal", "full"),
        required=True,
    )
    parser.add_argument("--launch-name", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--memory-probe", type=Path)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-image-digest", required=True)
    parser.add_argument(
        "--version-path",
        type=Path,
        default=Path("/opt/distillery/VERSION.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.child:
        print(
            "direct trainer invocation is forbidden; use distributed_launcher",
            file=sys.stderr,
        )
        return 2
    try:
        return run_child(args)
    except BaseException as exc:  # noqa: BLE001 - rank process must fail loudly
        print(f"qwen72b rank failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
