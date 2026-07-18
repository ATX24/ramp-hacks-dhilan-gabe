"""Network-isolated emergency QLoRA/LoRA trainer for pinned Qwen2.5 snapshots."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import random
import shutil
import signal
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

# These must be set before CUDA context creation. The actual seed is re-applied
# from the sealed manifest before model construction.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "17")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from distillery.contracts.hashing import content_sha256, sha256_hex
from distillery.contracts.manifest import SealedRunManifest
from distillery.training.models import (
    FrozenTeacherRuntimeGuard,
    assert_runtime_teacher_frozen,
    construct_optimizer_after_teacher_guard,
    freeze_and_assert_runtime_teacher,
)
from distillery.training.qlora import qlora_from_manifest_dict
from distillery.training.torch_losses import (
    forward_kl_chunked_torch,
    hard_cross_entropy_torch,
)
from experiments.aws_smoke.artifacts import (
    verify_emergency_artifacts,
    write_emergency_integrity,
)
from experiments.aws_smoke.channels import (
    discover_manifest,
    load_manifest,
)
from experiments.aws_smoke.deadline import Deadline, build_deadline
from experiments.aws_smoke.device_mapping import model_device_map
from experiments.aws_smoke.manifests import (
    build_sampler_plan,
    completion_provenance_sha256,
    manifest_arm,
    manifest_emergency_config,
    manifest_objective,
)
from experiments.aws_smoke.memory import (
    EmergencyMemoryProbeEvidence,
    select_precision_mode,
    validate_runtime_gpu_binding,
)
from experiments.aws_smoke.model_evidence import (
    TokenizerRuntimeEvidence,
    assert_loaded_tokenizers_compatible,
    require_local_model_weights,
    require_local_snapshot,
    verify_model_config_sha256,
    verify_tokenizer_runtime_evidence,
    write_tokenizer_evidence,
)
from experiments.aws_smoke.profile import RunArm
from experiments.aws_smoke.tokenization import (
    ArmTokenizationEvidence,
    TokenizedPair,
    build_chat_token_pair,
    build_prompt_ids,
    canonical_completion_records_sha256,
    completion_record_sha256,
)

SM_CHANNEL_MANIFEST = Path("/opt/ml/input/data/manifest")
SM_CHANNEL_DATASET = Path("/opt/ml/input/data/dataset")
SM_CHANNEL_MODELS = Path("/opt/ml/input/data/models")
SM_MODEL_DIR = Path("/opt/ml/model")
SM_OUTPUT_DIR = Path("/opt/ml/output/data")
SM_FAILURE_FILE = Path("/opt/ml/output/failure")


class TrainingCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TrainExample:
    example_id: str
    prompt_text: str
    target_text: str
    task: str
    difficulty: str
    record_sha256: str
    completion_record_sha256: str

def load_jsonl_examples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object on {path}:{line_number}")
        example_id = payload.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"missing example_id on {path}:{line_number}")
        if example_id in seen_ids:
            raise ValueError(f"duplicate example_id {example_id!r} in {path}")
        seen_ids.add(example_id)
        rows.append(payload)
    return rows


def render_prompt(example: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task": example.get("task"),
            "difficulty": example.get("difficulty"),
            "input": example.get("input"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def render_oracle_target(example: dict[str, Any]) -> str:
    target = example.get("expected_output")
    if not isinstance(target, dict) or not target:
        raise ValueError(f"example {example.get('example_id')} lacks oracle target")
    return json.dumps(target, sort_keys=True, ensure_ascii=False)


def load_teacher_response_map(
    path: Path | None,
    *,
    expected_sha256: str | None,
) -> dict[str, str]:
    if expected_sha256 is None:
        if path is not None:
            raise ValueError("teacher response path supplied without sealed hash")
        return {}
    if path is None or not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError("sealed sequence_kd requires teacher responses file")
    actual_sha256 = sha256_hex(path.read_bytes())
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "teacher responses hash mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    mapping: dict[str, str] = {}
    for row in load_jsonl_examples(path):
        example_id = str(row["example_id"])
        response = row.get("response_text")
        if not isinstance(response, str) or not response:
            raise ValueError(f"teacher response {example_id} is empty")
        mapping[example_id] = response
    return mapping


def materialize_examples(
    *,
    split_path: Path,
    arm: RunArm,
    teacher_responses: dict[str, str],
) -> list[TrainExample]:
    if not split_path.is_file():
        raise FileNotFoundError(f"missing split file: {split_path}")
    examples: list[TrainExample] = []
    for row in load_jsonl_examples(split_path):
        example_id = str(row["example_id"])
        if arm == "sequence_kd":
            try:
                target = teacher_responses[example_id]
            except KeyError:
                raise ValueError(
                    "sequence_kd missing pre-materialized teacher response for "
                    f"{example_id}"
                ) from None
        else:
            target = render_oracle_target(row)
        examples.append(
            TrainExample(
                example_id=example_id,
                prompt_text=render_prompt(row),
                target_text=target,
                task=str(row["task"]),
                difficulty=str(row["difficulty"]),
                record_sha256=content_sha256(row),
                completion_record_sha256=completion_record_sha256(
                    example_id=example_id,
                    target_text=target,
                    target_source=(
                        "pre_materialized_teacher"
                        if arm == "sequence_kd"
                        else "oracle"
                    ),
                ),
            )
        )
    if not examples:
        raise ValueError(f"split is empty: {split_path}")
    return examples


def set_determinism(seed: int) -> dict[str, Any]:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    unavoidable = [
        "GPU hardware and library-version changes can alter floating-point results",
    ]
    return {
        "seed": seed,
        "python_random_seeded": True,
        "numpy_seeded": True,
        "torch_seeded": True,
        "cuda_seeded": torch.cuda.is_available(),
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "tf32_disabled": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "unavoidable_gpu_nondeterminism": unavoidable,
        "bitwise_cross_hardware_reproducibility_claimed": False,
    }


def probe_bitsandbytes_nf4() -> tuple[bool, str]:
    """Run an actual tiny NF4 CUDA forward, not an import/CUDA-presence proxy."""
    if importlib.util.find_spec("bitsandbytes") is None:
        return False, "bitsandbytes module not installed"
    if not torch.cuda.is_available():
        return False, "CUDA unavailable"
    try:
        bnb = importlib.import_module("bitsandbytes")
        layer = bnb.nn.Linear4bit(
            16,
            16,
            bias=False,
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
            compress_statistics=True,
        )
        layer = layer.to("cuda")
        probe_input = torch.randn(
            2,
            16,
            device="cuda",
            dtype=torch.bfloat16,
        )
        with torch.inference_mode():
            output = layer(probe_input)
        if output.shape != (2, 16) or not bool(torch.isfinite(output).all().item()):
            return False, "NF4 probe returned malformed or non-finite output"
    except Exception as exc:  # noqa: BLE001 - kernel probe must report any runtime failure
        return False, f"{type(exc).__name__}: {exc}"
    return True, "tiny bitsandbytes Linear4bit NF4 CUDA forward passed"


def load_tokenizer(
    *,
    snapshot_dir: Path,
) -> Any:
    return AutoTokenizer.from_pretrained(
        str(snapshot_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )


def load_and_verify_tokenizers(
    manifest: SealedRunManifest,
    models_dir: Path,
) -> tuple[Any, TokenizerRuntimeEvidence, TokenizerRuntimeEvidence, Path, Path]:
    student_snapshot = require_local_snapshot(
        models_dir,
        manifest.models.student.id,
        manifest.models.student.revision,
    )
    teacher_snapshot = require_local_snapshot(
        models_dir,
        manifest.models.teacher.id,
        manifest.models.teacher.revision,
    )
    config = manifest_emergency_config(manifest)
    verify_model_config_sha256(
        student_snapshot,
        str(config["student_model_config_sha256"]),
    )
    verify_model_config_sha256(
        teacher_snapshot,
        str(config["teacher_model_config_sha256"]),
    )
    student_tokenizer = load_tokenizer(snapshot_dir=student_snapshot)
    teacher_tokenizer = load_tokenizer(snapshot_dir=teacher_snapshot)
    capability = manifest.training.qlora.capability_evidence
    if capability is None or capability.special_token_maps is None:
        raise ValueError("manifest lacks typed special-token map capability evidence")
    special_maps = capability.special_token_maps
    student_evidence = verify_tokenizer_runtime_evidence(
        snapshot_dir=student_snapshot,
        tokenizer=student_tokenizer,
        expected_tokenizer_sha256=manifest.models.student.tokenizer_sha256,
        expected_chat_template_sha256=manifest.models.student.chat_template_sha256,
        expected_special_token_map=dict(special_maps.student),
    )
    teacher_evidence = verify_tokenizer_runtime_evidence(
        snapshot_dir=teacher_snapshot,
        tokenizer=teacher_tokenizer,
        expected_tokenizer_sha256=manifest.models.teacher.tokenizer_sha256,
        expected_chat_template_sha256=manifest.models.teacher.chat_template_sha256,
        expected_special_token_map=dict(special_maps.teacher),
    )
    assert_loaded_tokenizers_compatible(teacher_evidence, student_evidence)
    return (
        student_tokenizer,
        student_evidence,
        teacher_evidence,
        student_snapshot,
        teacher_snapshot,
    )


def validate_lora_target_modules(
    model: Any,
    target_modules: Sequence[str],
) -> dict[str, int]:
    names = [name for name, _ in model.named_modules()]
    counts = {
        target: sum(
            1
            for name in names
            if name == target or name.endswith(f".{target}")
        )
        for target in target_modules
    }
    missing = sorted(target for target, count in counts.items() if count == 0)
    if missing or sum(counts.values()) == 0:
        raise RuntimeError(
            f"LoRA target modules matched no model modules for targets {missing or counts}"
        )
    return counts


def load_base_model(
    *,
    snapshot_dir: Path,
    precision_mode: str,
    for_training: bool,
) -> Any:
    require_local_model_weights(snapshot_dir)
    common: dict[str, Any] = {
        "pretrained_model_name_or_path": str(snapshot_dir),
        "local_files_only": True,
        "trust_remote_code": False,
        "device_map": model_device_map(torch),
    }
    if precision_mode == "qlora_nf4":
        common["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif precision_mode == "bf16_lora":
        common["torch_dtype"] = torch.bfloat16
    else:
        raise ValueError(f"unsupported precision mode {precision_mode!r}")
    base = AutoModelForCausalLM.from_pretrained(**common)
    if for_training:
        if precision_mode == "qlora_nf4":
            base = prepare_model_for_kbit_training(
                base,
                use_gradient_checkpointing=True,
            )
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base.enable_input_require_grads()
        base.config.use_cache = False
        if not bool(base.is_gradient_checkpointing):
            raise RuntimeError("student gradient checkpointing did not enable")
    return base


def load_student(
    *,
    snapshot_dir: Path,
    qlora_dict: dict[str, Any],
    max_length: int,
    load_teacher: bool,
) -> tuple[Any, dict[str, Any]]:
    sealed_mode = str(qlora_dict["precision_mode"])
    nf4_passed, nf4_detail = probe_bitsandbytes_nf4()
    raw_memory_evidence = qlora_dict.get("memory_probe_evidence")
    memory_evidence = (
        EmergencyMemoryProbeEvidence.model_validate(raw_memory_evidence)
        if raw_memory_evidence is not None
        else None
    )
    if memory_evidence is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("sealed GPU memory evidence requires an available CUDA device")
        properties = torch.cuda.get_device_properties(0)
        validate_runtime_gpu_binding(
            memory_evidence,
            device_type=torch.cuda.get_device_name(0),
            capacity_memory_bytes=int(properties.total_memory),
        )
    estimate = select_precision_mode(
        sealed_mode=sealed_mode,  # type: ignore[arg-type]
        nf4_kernel_probe_passed=nf4_passed,
        bf16_memory_evidence=memory_evidence,
        max_length=max_length,
        microbatch=int(qlora_dict["microbatch"]),
        lora_rank=int(qlora_dict["rank"]),
        load_teacher=load_teacher,
    )
    base = load_base_model(
        snapshot_dir=snapshot_dir,
        precision_mode=estimate.mode,
        for_training=True,
    )
    adapter_config = {
        key: qlora_dict[key]
        for key in (
            "rank",
            "alpha",
            "dropout",
            "target_modules",
            "bias",
            "task_type",
            "use_rslora",
            "modules_to_save",
        )
    }
    qlora_config = qlora_from_manifest_dict(adapter_config)
    module_matches = validate_lora_target_modules(
        base,
        qlora_config.target_modules,
    )
    student = get_peft_model(base, LoraConfig(**qlora_config.to_peft_dict()))
    student.train()
    if not any(parameter.requires_grad for parameter in student.parameters()):
        raise RuntimeError("LoRA student has no trainable parameters")
    metadata = {
        "precision_mode": estimate.mode,
        "deviation_label": estimate.deviation_label,
        "protocol_deviation": qlora_dict.get("protocol_deviation"),
        "memory_estimate_bytes": estimate.total_bytes,
        "nf4_kernel_probe_passed": nf4_passed,
        "nf4_kernel_probe_detail": nf4_detail,
        "bf16_memory_evidence": raw_memory_evidence,
        "gradient_checkpointing_enabled": bool(student.base_model.is_gradient_checkpointing),
        "lora_target_module_matches": module_matches,
    }
    if estimate.mode == "qlora_nf4" and not nf4_passed:
        raise RuntimeError("QLoRA proceeded despite failed NF4 kernel probe")
    if estimate.mode == "bf16_lora" and estimate.deviation_label is None:
        raise RuntimeError("BF16 LoRA proceeded without a protocol deviation label")
    return student, metadata


def load_teacher(
    *,
    snapshot_dir: Path,
) -> tuple[Any, FrozenTeacherRuntimeGuard]:
    require_local_model_weights(snapshot_dir)
    teacher = AutoModelForCausalLM.from_pretrained(
        str(snapshot_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map=model_device_map(torch),
    )
    guard = freeze_and_assert_runtime_teacher(teacher)
    return teacher, guard


def tokenize_pair(
    tokenizer: Any,
    prompt: str,
    target: str,
    *,
    max_length: int,
    max_completion: int,
) -> dict[str, Any]:
    pair = build_chat_token_pair(
        tokenizer,
        prompt,
        target,
        max_length=max_length,
        max_completion=max_completion,
    )
    return {
        "input_ids": torch.tensor([pair.input_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(pair.input_ids)), dtype=torch.long),
        "labels": torch.tensor([pair.labels], dtype=torch.long),
        "completion_mask": torch.tensor(
            [pair.completion_mask],
            dtype=torch.float32,
        ),
        "token_evidence": pair,
    }


def shift_for_lm(
    logits: torch.Tensor,
    labels: torch.Tensor,
    completion_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.shape[:-1] != labels.shape or labels.shape != completion_mask.shape:
        raise ValueError("logits, labels, and completion mask shapes are incompatible")
    if logits.shape[1] < 2:
        raise ValueError("causal LM sequence must contain at least two tokens")
    return logits[:, :-1, :], labels[:, 1:], completion_mask[:, 1:]


def compute_torch_loss(
    *,
    arm: RunArm,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    completion_mask: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    temperature: float,
    kd_weight: float,
    hard_ce_weight: float,
    vocab_chunk: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Production objective using shared Distillery CE and exact full-vocab KL."""
    if abs(kd_weight + hard_ce_weight - 1.0) > 1e-9:
        raise ValueError("kd_weight + hard_ce_weight must equal 1")
    student_shifted, labels_shifted, mask_shifted = shift_for_lm(
        student_logits,
        labels,
        completion_mask,
    )
    student_for_loss = student_shifted.float()
    if arm in {"oracle_sft", "ce_ablation", "sequence_kd"}:
        if teacher_logits is not None:
            raise RuntimeError(f"{arm} must omit online teacher runtime")
        ce = hard_cross_entropy_torch(
            student_for_loss,
            labels_shifted,
            position_mask=mask_shifted,
            ignore_index=-100,
        )
        return ce, {
            "loss": float(ce.detach().item()),
            "ce": float(ce.detach().item()),
            "kl": 0.0,
            "kd_term": 0.0,
        }
    if arm != "logit_kd":
        raise ValueError(f"unsupported arm: {arm}")
    if teacher_logits is None:
        raise RuntimeError("logit_kd requires frozen teacher logits; refusing fake KD")
    if teacher_logits.requires_grad:
        raise RuntimeError("teacher logits require gradients; teacher is not frozen")
    teacher_shifted, _, teacher_mask = shift_for_lm(
        teacher_logits,
        labels,
        completion_mask,
    )
    if teacher_shifted.shape != student_shifted.shape:
        raise RuntimeError(
            "teacher/student full-vocabulary logit shapes differ: "
            f"{teacher_shifted.shape} vs {student_shifted.shape}"
        )
    if not torch.equal(teacher_mask, mask_shifted):
        raise RuntimeError("teacher/student completion masks diverged")
    kl = forward_kl_chunked_torch(
        teacher_shifted.float(),
        student_for_loss,
        temperature=temperature,
        vocab_chunk_size=vocab_chunk,
        position_mask=mask_shifted,
    )
    ce = hard_cross_entropy_torch(
        student_for_loss,
        labels_shifted,
        position_mask=mask_shifted,
        ignore_index=-100,
    )
    kd_term = kl * (temperature * temperature)
    loss = kd_weight * kd_term + hard_ce_weight * ce
    return loss, {
        "loss": float(loss.detach().item()),
        "ce": float(ce.detach().item()),
        "kl": float(kl.detach().item()),
        "kd_term": float(kd_term.detach().item()),
    }


def verify_and_order_training_examples(
    *,
    examples: list[TrainExample],
    tokenizer: Any,
    manifest: SealedRunManifest,
    source_file_sha256: str,
) -> tuple[list[TrainExample], dict[str, TokenizedPair]]:
    qlora = manifest.training.qlora
    completion_evidence = manifest.training.completion_evidence
    if completion_evidence is None:
        raise ValueError("manifest lacks typed completion evidence")
    config = manifest_emergency_config(manifest)
    max_completion = int(qlora.max_completion)
    pairs: dict[str, TokenizedPair] = {}
    for example in examples:
        pairs[example.example_id] = build_chat_token_pair(
            tokenizer,
            example.prompt_text,
            example.target_text,
            max_length=manifest.training.max_length,
            max_completion=max_completion,
        )
    counts = {
        example_id: pair.completion_token_count
        for example_id, pair in pairs.items()
    }
    originals = {
        example_id: pair.original_completion_token_count
        for example_id, pair in pairs.items()
    }
    prompt_counts = {
        example_id: pair.prompt_token_count
        for example_id, pair in pairs.items()
    }
    total_counts = {
        example_id: len(pair.input_ids)
        for example_id, pair in pairs.items()
    }
    record_hashes = {
        example.example_id: example.record_sha256 for example in examples
    }
    completion_record_hashes = {
        example.example_id: example.completion_record_sha256 for example in examples
    }
    expected_counts = {
        str(key): int(value)
        for key, value in completion_evidence.completion_token_counts.items()
    }
    if counts != expected_counts:
        raise ValueError(
            "actual tokenizer-derived completion counts differ from sealed evidence"
        )
    if source_file_sha256 != completion_evidence.source_file_sha256:
        raise ValueError("actual completion source-file hash differs from sealed evidence")
    if completion_record_hashes != dict(completion_evidence.record_sha256):
        raise ValueError("actual completion record hashes differ from sealed evidence")
    canonical_records = canonical_completion_records_sha256(
        completion_record_hashes
    )
    if canonical_records != completion_evidence.canonical_records_sha256:
        raise ValueError("actual canonical completion-record hash differs from evidence")
    actual_truncated = sorted(
        example_id for example_id, pair in pairs.items() if pair.completion_truncated
    )
    arm = manifest_arm(manifest)
    runtime_tokenization = ArmTokenizationEvidence(
        arm=arm,
        target_source=(
            "pre_materialized_teacher" if arm == "sequence_kd" else "oracle"
        ),
        completion_token_counts=counts,
        prompt_token_counts=prompt_counts,
        total_token_counts=total_counts,
        record_sha256=record_hashes,
        source_file_sha256=source_file_sha256,
        canonical_records_sha256=canonical_records,
        completion_record_sha256=completion_record_hashes,
        original_completion_token_counts=originals,
        truncated_example_ids=tuple(actual_truncated),
        teacher_responses_sha256=(
            str(config["teacher_responses_sha256"])
            if arm == "sequence_kd" and config["teacher_responses_sha256"] is not None
            else None
        ),
    )
    if (
        completion_provenance_sha256(runtime_tokenization)
        != completion_evidence.provenance_sha256
    ):
        raise ValueError(
            "actual prompt/total/original token counts, truncation, or record "
            "hashes differ from sealed completion provenance"
        )

    plan = build_sampler_plan(
        example_ids=[example.example_id for example in examples],
        tasks=[example.task for example in examples],
        difficulties=[example.difficulty for example in examples],
        completion_token_counts=counts,
        prompt_token_counts=prompt_counts,
        total_token_counts=total_counts,
        record_sha256=record_hashes,
        seed=manifest.training.seed,
        tokenizer_sha256=manifest.models.student.tokenizer_sha256,
        microbatch_size=int(config["microbatch"]),
    )
    if plan.sampler_order_hash != manifest.sampler_order_hash:
        raise ValueError(
            "trainer-recomputed sampler_order_hash differs from sealed manifest"
        )
    by_id = {example.example_id: example for example in examples}
    return [by_id[example_id] for example_id in plan.order], pairs


def hash_trainable_initialization(model: Any) -> str:
    digest = hashlib.sha256()
    trainable = 0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        trainable += 1
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    if trainable == 0:
        raise RuntimeError("cannot hash initialization: no trainable parameters")
    return digest.hexdigest()


def write_predictions(
    *,
    student: Any,
    tokenizer: Any,
    validation_examples: list[TrainExample],
    path: Path,
    max_length: int,
    max_completion: int,
    deadline: Deadline,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    student.eval()
    previous_use_cache = bool(student.config.use_cache)
    student.config.use_cache = True
    count = 0
    try:
        with path.open("w", encoding="utf-8") as handle:
            for example in validation_examples:
                deadline.require_finalize_time(f"prediction {example.example_id}")
                prompt_ids = build_prompt_ids(tokenizer, example.prompt_text)
                if len(prompt_ids) + max_completion > max_length:
                    raise ValueError(
                        f"validation prompt {example.example_id} leaves no sealed "
                        "completion budget"
                    )
                device = next(student.parameters()).device
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                with torch.inference_mode():
                    generated = student.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        max_new_tokens=max_completion,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = generated[0, len(prompt_ids) :].detach().cpu().tolist()
                prediction = tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                )
                handle.write(
                    json.dumps(
                        {
                            "example_id": example.example_id,
                            "task": example.task,
                            "difficulty": example.difficulty,
                            "prediction_text": prediction,
                            "generation": "greedy",
                            "generated_token_count": len(new_tokens),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                count += 1
    finally:
        student.config.use_cache = previous_use_cache
    return count


def reload_adapter_for_test(
    *,
    snapshot_dir: Path,
    adapter_dir: Path,
    precision_mode: str,
    pair: TokenizedPair,
    deadline: Deadline,
) -> dict[str, Any]:
    deadline.require_finalize_time("fresh adapter reload")
    fresh_base = load_base_model(
        snapshot_dir=snapshot_dir,
        precision_mode=precision_mode,
        for_training=False,
    )
    reloaded = PeftModel.from_pretrained(
        fresh_base,
        str(adapter_dir),
        is_trainable=False,
        local_files_only=True,
    )
    reloaded.eval()
    device = next(reloaded.parameters()).device
    input_ids = torch.tensor([pair.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = reloaded(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    finite = bool(torch.isfinite(output.logits).all().item())
    report = {
        "passed": finite,
        "fresh_base_loaded": True,
        "adapter_reloaded": True,
        "forward_finite": finite,
        "adapter_path": str(adapter_dir),
        "precision_mode": precision_mode,
    }
    del reloaded
    del fresh_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not finite:
        raise RuntimeError("fresh adapter reload produced non-finite logits")
    return report


def run_training(
    *,
    manifest: SealedRunManifest,
    arm: RunArm,
    dataset_dir: Path,
    models_dir: Path,
    output_dir: Path,
    model_output_dir: Path = SM_MODEL_DIR,
    teacher_responses_path: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    qlora = manifest.training.qlora
    config = manifest_emergency_config(manifest)
    if manifest_arm(manifest) != arm:
        raise ValueError(f"CLI arm {arm} != sealed manifest arm {manifest_arm(manifest)}")
    if not bool(config["gradient_checkpointing"]):
        raise ValueError("sealed gradient checkpointing must be enabled")
    if not bool(config["deterministic_algorithms"]):
        raise ValueError("sealed deterministic algorithms must be enabled")
    microbatch = int(config["microbatch"])
    if microbatch != 1:
        raise ValueError("emergency runtime currently supports sealed microbatch=1 only")
    grad_accumulation = int(config["grad_accumulation"])
    if grad_accumulation < 1:
        raise ValueError("grad_accumulation must be >= 1")
    max_completion = int(qlora.max_completion)
    if max_completion < 1 or max_completion >= manifest.training.max_length:
        raise ValueError("sealed max_completion is invalid")
    if not 5 <= manifest.training.max_steps <= 10:
        raise ValueError("emergency trainer requires 5-10 optimizer steps")

    tagged_runtime_seconds = int(manifest.tags["MaxRuntimeInSeconds"])
    configured_runtime_seconds = int(config["max_runtime_seconds"])
    if tagged_runtime_seconds != configured_runtime_seconds:
        raise ValueError(
            "sealed MaxRuntimeInSeconds tag differs from EmergencyConfig runtime"
        )
    deadline = build_deadline(
        max_runtime_seconds=tagged_runtime_seconds,
        artifact_reserve_seconds=int(config["artifact_reserve_seconds"]),
        shutdown_margin_seconds=int(config["shutdown_margin_seconds"]),
        clock=clock,
    )
    determinism = set_determinism(manifest.training.seed)
    if config["precision_mode"] == "qlora_nf4":
        determinism["unavoidable_gpu_nondeterminism"].append(
            "bitsandbytes NF4 kernels are not claimed bitwise deterministic"
        )

    response_path = teacher_responses_path
    if response_path is None and arm == "sequence_kd":
        response_path = dataset_dir / "teacher_responses.jsonl"
    teacher_responses = load_teacher_response_map(
        response_path,
        expected_sha256=(
            str(config["teacher_responses_sha256"])
            if arm == "sequence_kd" and config["teacher_responses_sha256"] is not None
            else None
        ),
    )
    train_path = dataset_dir / "train.jsonl"
    validation_path = dataset_dir / "validation.jsonl"
    actual_train_sha256 = sha256_hex(train_path.read_bytes())
    actual_validation_sha256 = sha256_hex(validation_path.read_bytes())
    if actual_train_sha256 != manifest.dataset.split_sha256["train"]:
        raise ValueError("train split hash differs from sealed dataset evidence")
    if actual_validation_sha256 != manifest.dataset.split_sha256["validation"]:
        raise ValueError("validation split hash differs from sealed dataset evidence")
    source_file_sha256 = (
        sha256_hex(response_path.read_bytes())
        if arm == "sequence_kd" and response_path is not None
        else actual_train_sha256
    )
    train_examples = materialize_examples(
        split_path=train_path,
        arm=arm,
        teacher_responses=teacher_responses,
    )
    validation_examples = materialize_examples(
        split_path=validation_path,
        arm="oracle_sft",
        teacher_responses={},
    )
    (
        tokenizer,
        student_tokenizer_evidence,
        teacher_tokenizer_evidence,
        student_snapshot,
        teacher_snapshot,
    ) = load_and_verify_tokenizers(manifest, models_dir)
    ordered_examples, token_pairs = verify_and_order_training_examples(
        examples=train_examples,
        tokenizer=tokenizer,
        manifest=manifest,
        source_file_sha256=source_file_sha256,
    )
    deadline.require_training_time("model load")
    print(
        json.dumps(
            {
                "event": "model_load_start",
                "student_model_id": manifest.models.student.id,
                "student_revision": manifest.models.student.revision,
                "teacher_required": arm == "logit_kd",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    teacher_required = arm == "logit_kd"
    runtime_qlora = qlora.model_dump(mode="json")
    runtime_qlora.update(config)
    student, precision_metadata = load_student(
        snapshot_dir=student_snapshot,
        qlora_dict=runtime_qlora,
        max_length=manifest.training.max_length,
        load_teacher=teacher_required,
    )
    initialization_sha256 = hash_trainable_initialization(student)
    teacher = None
    teacher_guard = None
    teacher_load_seconds = 0.0
    if teacher_required:
        teacher_load_started = clock()
        teacher, teacher_guard = load_teacher(snapshot_dir=teacher_snapshot)
        teacher_load_seconds = clock() - teacher_load_started

    learning_rate = float(config["learning_rate"])
    if teacher is not None:
        optimizer = construct_optimizer_after_teacher_guard(
            lambda parameters: torch.optim.AdamW(parameters, lr=learning_rate),
            (parameter for parameter in student.parameters() if parameter.requires_grad),
            teacher=teacher,
            teacher_guard=teacher_guard,
        )
    else:
        optimizer = torch.optim.AdamW(
            (parameter for parameter in student.parameters() if parameter.requires_grad),
            lr=learning_rate,
        )
    print(
        json.dumps(
            {
                "event": "optimizer_ready",
                "arm": arm,
                "optimizer": "torch.optim.AdamW",
                "student_model_loaded": True,
                "teacher_model_loaded": teacher is not None,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    temperature = float(qlora.logit_temperature)
    kd_weight = float(qlora.kd_weight)
    hard_ce_weight = float(qlora.hard_ce_weight)
    vocab_chunk = int(qlora.vocab_chunk)
    max_steps = manifest.training.max_steps
    metrics_path = output_dir / "training/metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cursor = 0
    teacher_forward_seconds = 0.0
    completed_steps = 0
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, max_steps + 1):
            deadline.require_training_time(f"optimizer step {step}")
            optimizer.zero_grad(set_to_none=True)
            aggregate = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "kd_term": 0.0}
            example_ids: list[str] = []
            completion_counts: list[int] = []
            for accumulation_index in range(grad_accumulation):
                deadline.require_training_time(
                    f"step {step} accumulation {accumulation_index + 1}"
                )
                example = ordered_examples[cursor % len(ordered_examples)]
                cursor += 1
                example_ids.append(example.example_id)
                pair = token_pairs[example.example_id]
                completion_counts.append(pair.completion_token_count)
                batch = tokenize_pair(
                    tokenizer,
                    example.prompt_text,
                    example.target_text,
                    max_length=manifest.training.max_length,
                    max_completion=max_completion,
                )
                device = next(student.parameters()).device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                completion_mask = batch["completion_mask"].to(device)
                student_output = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                teacher_logits = None
                if teacher is not None:
                    assert_runtime_teacher_frozen(teacher, guard=teacher_guard)
                    teacher_started = clock()
                    with torch.inference_mode():
                        teacher_output = teacher(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        teacher_logits = teacher_output.logits.detach()
                    teacher_forward_seconds += clock() - teacher_started
                loss, parts = compute_torch_loss(
                    arm=arm,
                    student_logits=student_output.logits,
                    labels=labels,
                    completion_mask=completion_mask,
                    teacher_logits=teacher_logits,
                    temperature=temperature,
                    kd_weight=kd_weight,
                    hard_ce_weight=hard_ce_weight,
                    vocab_chunk=vocab_chunk,
                )
                (loss / grad_accumulation).backward()
                for key in aggregate:
                    aggregate[key] += parts[key] / grad_accumulation
            optimizer.step()
            completed_steps = step
            metrics_file.write(
                json.dumps(
                    {
                        "step": step,
                        "arm": arm,
                        "example_ids": example_ids,
                        "microbatch": microbatch,
                        "grad_accumulation": grad_accumulation,
                        "completion_token_counts": completion_counts,
                        "teacher_forward_seconds_cumulative": teacher_forward_seconds,
                        **aggregate,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            metrics_file.flush()

    deadline.require_finalize_time("adapter save")
    adapter_dir = output_dir / "model/adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights.is_file() or adapter_weights.stat().st_size == 0:
        raise RuntimeError("PEFT save did not produce nonempty adapter_model.safetensors")
    write_tokenizer_evidence(
        output_dir / "model/tokenizer_evidence.json",
        student=student_tokenizer_evidence,
        teacher=teacher_tokenizer_evidence,
    )
    chat_template = tokenizer.chat_template
    if not isinstance(chat_template, str) or not chat_template.strip():
        raise RuntimeError("loaded tokenizer chat template disappeared before save")
    (output_dir / "model/chat_template.txt").write_text(
        chat_template,
        encoding="utf-8",
    )

    prediction_count = write_predictions(
        student=student,
        tokenizer=tokenizer,
        validation_examples=validation_examples,
        path=output_dir / "evaluation/predictions.jsonl",
        max_length=manifest.training.max_length,
        max_completion=max_completion,
        deadline=deadline,
    )

    # Release treatment-only teacher and trained model before a genuinely fresh
    # base + adapter reload.
    del optimizer
    if teacher is not None:
        del teacher
    del student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    first_pair = token_pairs[ordered_examples[0].example_id]
    load_test = reload_adapter_for_test(
        snapshot_dir=student_snapshot,
        adapter_dir=adapter_dir,
        precision_mode=str(config["precision_mode"]),
        pair=first_pair,
        deadline=deadline,
    )
    load_test_path = output_dir / "model/load_test.json"
    load_test_path.write_text(
        json.dumps(load_test, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(adapter_dir, model_output_dir, dirs_exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    elapsed_seconds = clock() - deadline.started_monotonic
    child_runtime_cost = (
        float(manifest.tags["HourlyUsd"]) * elapsed_seconds / 3600.0
    )
    shared_campaign = os.environ.get("DISTILLERY_SHARED_CAMPAIGN") == "1"
    gross_cost = 0.0 if shared_campaign else child_runtime_cost
    cost_payload = {
        "schema_version": "distillery.aws_smoke.cost.v1",
        "elapsed_seconds": elapsed_seconds,
        "hourly_usd": float(manifest.tags["HourlyUsd"]),
        "gross_cost_usd": gross_cost,
        "max_run_usd": manifest.cost.max_run_usd,
        "cost_kind": (
            "nonbillable_campaign_child_observation"
            if shared_campaign
            else "runtime_estimate_from_elapsed_wall_clock"
        ),
        "child_runtime_observation_usd": child_runtime_cost,
        "parent_campaign_allocation_authoritative": shared_campaign,
        "teacher_load_seconds": teacher_load_seconds,
        "teacher_forward_seconds": teacher_forward_seconds,
        "teacher_treatment_overhead_included": teacher_required,
    }
    cost_path = output_dir / "costs/gross_cost.json"
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    cost_path.write_text(
        json.dumps(cost_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    objective = manifest_objective(manifest)
    emergency_run = {
        "schema_version": "distillery.aws_smoke.run_report.v2",
        "status": "completed",
        "arm": arm,
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.seal_sha256(),
        "max_steps": max_steps,
        "completed_steps": completed_steps,
        "seed": manifest.training.seed,
        "student_revision": manifest.models.student.revision,
        "initialization_fingerprint": manifest.tags["InitializationFingerprint"],
        "adapter_initialization_sha256": initialization_sha256,
        "precision": precision_metadata,
        "determinism": determinism,
        "sampler_order_hash": manifest.sampler_order_hash,
        "sampler_order": [example.example_id for example in ordered_examples],
        "objective": objective,
        "hard_target_equivalent_to": objective["equivalent_to"],
        "distinct_training_signal": objective["distinct_training_signal"],
        "teacher_runtime": objective["teacher_runtime"],
        "teacher_load_seconds": teacher_load_seconds,
        "teacher_forward_seconds": teacher_forward_seconds,
        "fake_kd": False,
        "full_vocabulary_kl": arm == "logit_kd",
        "prediction_count": prediction_count,
        "adapter_reload_passed": load_test["passed"],
        "elapsed_seconds": elapsed_seconds,
    }
    emergency_path = output_dir / "training/emergency_run.json"
    emergency_path.write_text(
        json.dumps(emergency_run, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    deadline.require_finalize_time("artifact checksums")
    write_emergency_integrity(output_dir)
    artifact_report = verify_emergency_artifacts(output_dir)
    return {**emergency_run, "artifact_report": artifact_report}


def write_failure_output(
    *,
    output_dir: Path,
    failure_file: Path,
    error: BaseException,
    cancelled: bool,
) -> None:
    payload = {
        "schema_version": "distillery.aws_smoke.failure.v1",
        "status": "cancelled" if cancelled else "failed",
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "failure.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        failure_file.parent.mkdir(parents=True, exist_ok=True)
        failure_file.write_text(
            f"{payload['status']}: {payload['error_type']}: {payload['message']}\n",
            encoding="utf-8",
        )
    except OSError:
        # Preserve the original training exception if disk failure prevents
        # SageMaker failure-file creation.
        return


def _raise_cancelled(signum: int, _frame: Any) -> None:
    raise TrainingCancelled(f"received cancellation signal {signum}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="experiments.aws_smoke.train")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--arm",
        required=True,
        choices=["oracle_sft", "ce_ablation", "logit_kd", "sequence_kd"],
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-output-dir", type=Path, default=None)
    parser.add_argument("--teacher-responses", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = stdout or sys.stdout
    manifest_channel = SM_CHANNEL_MANIFEST
    dataset_dir = args.dataset_dir or SM_CHANNEL_DATASET
    models_dir = args.models_dir or SM_CHANNEL_MODELS
    output_dir = args.output_dir or Path(
        os.environ.get("SM_OUTPUT_DATA_DIR", str(SM_OUTPUT_DIR))
    )
    model_output_dir = args.model_output_dir or Path(
        os.environ.get("SM_MODEL_DIR", str(SM_MODEL_DIR))
    )
    failure_file = Path(
        os.environ.get("DISTILLERY_FAILURE_PATH", str(SM_FAILURE_FILE))
    )
    previous_term = signal.signal(signal.SIGTERM, _raise_cancelled)
    previous_int = signal.signal(signal.SIGINT, _raise_cancelled)
    try:
        manifest_path = args.manifest or discover_manifest(manifest_channel)
        manifest = load_manifest(manifest_path)
        print(
            json.dumps(
                {
                    "event": "emergency_trainer_start",
                    "arm": args.arm,
                    "manifest_sha256": manifest.seal_sha256(),
                    "run_id": manifest.run_id,
                },
                sort_keys=True,
            ),
            file=out,
            flush=True,
        )
        result = run_training(
            manifest=manifest,
            arm=args.arm,
            dataset_dir=dataset_dir,
            models_dir=models_dir,
            output_dir=output_dir,
            model_output_dir=model_output_dir,
            teacher_responses_path=args.teacher_responses,
        )
    except BaseException as exc:
        cancelled = isinstance(exc, (KeyboardInterrupt, TrainingCancelled))
        write_failure_output(
            output_dir=output_dir,
            failure_file=failure_file,
            error=exc,
            cancelled=cancelled,
        )
        print(
            json.dumps(
                {
                    "ok": False,
                    "status": "cancelled" if cancelled else "failed",
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=out,
        )
        return 2 if cancelled else 1
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
    print(json.dumps({"ok": True, **result}, sort_keys=True), file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
