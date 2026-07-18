"""Container-compatible emergency trainer for pinned Qwen2.5 student arms.

Consumes SageMaker File-mode channels and a sealed emergency manifest. Implements
oracle_sft / ce_ablation / logit_kd with exact full-vocabulary forward KL at
completion positions for logit_kd. sequence_kd is optional and requires
pre-materialized teacher responses.

This module imports ML stacks at top level on purpose. If those imports fail,
the emergency path is not ready — do not claim otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from distillery.contracts.manifest import SealedRunManifest
from distillery.training.models import (
    FrozenTeacherRuntimeGuard,
    construct_optimizer_after_teacher_guard,
    freeze_and_assert_runtime_teacher,
)
from distillery.training.qlora import qlora_from_manifest_dict
from distillery.training.torch_losses import (
    forward_kl_chunked_torch,
    hard_cross_entropy_torch,
)
from experiments.aws_smoke.artifacts import write_emergency_integrity
from experiments.aws_smoke.memory import select_precision_mode
from experiments.aws_smoke.profile import RunArm

SM_CHANNEL_MANIFEST = Path("/opt/ml/input/data/manifest")
SM_CHANNEL_DATASET = Path("/opt/ml/input/data/dataset")
SM_CHANNEL_MODELS = Path("/opt/ml/input/data/models")
SM_MODEL_DIR = Path("/opt/ml/model")
SM_OUTPUT_DIR = Path("/opt/ml/output")


@dataclass(frozen=True, slots=True)
class TrainExample:
    example_id: str
    prompt_text: str
    target_text: str
    task: str
    difficulty: str


def _resolve_path(path: Path | None, default: Path) -> Path:
    return path if path is not None else default


def load_manifest(path: Path) -> SealedRunManifest:
    return SealedRunManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def discover_manifest(channel: Path) -> Path:
    direct = channel / "manifest.json"
    if direct.is_file():
        return direct
    matches = sorted(channel.glob("manifest_*.json"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"no unique manifest in {channel}")


def load_jsonl_examples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
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
    return json.dumps(example.get("expected_output") or {}, sort_keys=True, ensure_ascii=False)


def load_teacher_response_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    mapping: dict[str, str] = {}
    for row in load_jsonl_examples(path):
        example_id = str(row["example_id"])
        mapping[example_id] = str(row["response_text"])
    return mapping


def materialize_examples(
    *,
    dataset_dir: Path,
    arm: RunArm,
    teacher_responses: dict[str, str],
) -> list[TrainExample]:
    train_path = dataset_dir / "train.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"missing train.jsonl in dataset channel: {train_path}")
    examples: list[TrainExample] = []
    for row in load_jsonl_examples(train_path):
        example_id = str(row["example_id"])
        if arm == "sequence_kd":
            if example_id not in teacher_responses:
                raise ValueError(
                    f"sequence_kd missing pre-materialized teacher response for {example_id}"
                )
            target = teacher_responses[example_id]
        else:
            target = render_oracle_target(row)
        examples.append(
            TrainExample(
                example_id=example_id,
                prompt_text=render_prompt(row),
                target_text=target,
                task=str(row["task"]),
                difficulty=str(row["difficulty"]),
            )
        )
    if not examples:
        raise ValueError("train split is empty")
    return examples


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _bitsandbytes_status() -> tuple[bool, bool]:
    try:
        import bitsandbytes as bnb  # noqa: F401
    except Exception:
        return False, False
    # Reliability: CUDA required for NF4 training path in this emergency profile.
    reliable = bool(torch.cuda.is_available())
    return True, reliable


def resolve_model_local_path(models_dir: Path, model_id: str, revision: str) -> Path | None:
    """Optional local snapshot layout: models/<org>/<name>/<revision>/."""
    org, _, name = model_id.partition("/")
    if not org or not name:
        return None
    candidate = models_dir / org / name / revision
    if candidate.is_dir():
        return candidate
    return None


def load_tokenizer(model_id: str, revision: str, models_dir: Path) -> Any:
    local = resolve_model_local_path(models_dir, model_id, revision)
    source = str(local) if local is not None else model_id
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=None if local is not None else revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_student(
    *,
    model_id: str,
    revision: str,
    models_dir: Path,
    qlora_dict: dict[str, Any],
    max_length: int,
    load_teacher: bool,
) -> tuple[Any, dict[str, Any]]:
    available, reliable = _bitsandbytes_status()
    estimate = select_precision_mode(
        bitsandbytes_available=available,
        bitsandbytes_reliable=reliable,
        max_length=max_length,
        microbatch=int(qlora_dict.get("microbatch", 1)),
        lora_rank=int(qlora_dict.get("rank", 8)),
        load_teacher=load_teacher,
    )
    local = resolve_model_local_path(models_dir, model_id, revision)
    source = str(local) if local is not None else model_id
    revision_kw = None if local is not None else revision
    qlora_cfg = qlora_from_manifest_dict(qlora_dict)
    meta: dict[str, Any] = {
        "precision_mode": estimate.mode,
        "deviation_label": estimate.deviation_label,
        "memory_estimate_bytes": estimate.total_bytes,
        "bitsandbytes_available": available,
        "bitsandbytes_reliable": reliable,
        "source": source,
    }
    if estimate.mode == "qlora_nf4":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            source,
            revision=revision_kw,
            quantization_config=quant,
            device_map="auto",
            trust_remote_code=False,
        )
        base = prepare_model_for_kbit_training(base)
    else:
        if estimate.deviation_label is None:
            raise RuntimeError("bf16_lora selected without deviation label")
        base = AutoModelForCausalLM.from_pretrained(
            source,
            revision=revision_kw,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
        )
    lora = LoraConfig(**qlora_cfg.to_peft_dict())
    student = get_peft_model(base, lora)
    student.train()
    return student, meta


def load_teacher(
    *,
    model_id: str,
    revision: str,
    models_dir: Path,
) -> tuple[Any, FrozenTeacherRuntimeGuard]:
    local = resolve_model_local_path(models_dir, model_id, revision)
    source = str(local) if local is not None else model_id
    teacher = AutoModelForCausalLM.from_pretrained(
        source,
        revision=None if local is not None else revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    guard = freeze_and_assert_runtime_teacher(teacher)
    return teacher, guard


def tokenize_pair(
    tokenizer: Any,
    prompt: str,
    target: str,
    *,
    max_length: int,
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    # Causal LM: concatenate prompt + target; predict target tokens only.
    input_ids = (prompt_ids + target_ids)[:max_length]
    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    attention = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "completion_mask": torch.tensor(
            [[0.0] * prompt_len + [1.0] * (len(input_ids) - prompt_len)],
            dtype=torch.float32,
        ),
    }


def shift_for_lm(
    logits: torch.Tensor,
    labels: torch.Tensor,
    completion_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Standard next-token shift.
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
    s_logits, s_labels, s_mask = shift_for_lm(student_logits, labels, completion_mask)
    if arm in {"oracle_sft", "ce_ablation", "sequence_kd"}:
        loss = hard_cross_entropy_torch(
            s_logits,
            s_labels,
            position_mask=s_mask,
            ignore_index=-100,
        )
        return loss, {
            "loss": float(loss.detach().item()),
            "ce": float(loss.detach().item()),
            "kl": 0.0,
        }
    if arm == "logit_kd":
        if teacher_logits is None:
            raise RuntimeError("logit_kd requires frozen teacher logits; refusing fake KD")
        t_logits, _, t_mask = shift_for_lm(teacher_logits, labels, completion_mask)
        if t_logits.shape != s_logits.shape:
            raise RuntimeError(
                f"teacher/student logit shape mismatch: {t_logits.shape} vs {s_logits.shape}"
            )
        if not torch.equal(t_mask, s_mask):
            raise RuntimeError("teacher/student completion masks diverged")
        kl = forward_kl_chunked_torch(
            t_logits,
            s_logits,
            temperature=temperature,
            vocab_chunk_size=vocab_chunk,
            position_mask=s_mask,
        )
        ce = hard_cross_entropy_torch(
            s_logits,
            s_labels,
            position_mask=s_mask,
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
    raise ValueError(f"unsupported arm: {arm}")


def run_training(
    *,
    manifest: SealedRunManifest,
    arm: RunArm,
    dataset_dir: Path,
    models_dir: Path,
    output_dir: Path,
    teacher_responses_path: Path | None = None,
) -> dict[str, Any]:
    if arm != str(manifest.training.qlora.get("arm", arm)):
        # Allow CLI override only when manifest arm matches or is absent.
        manifest_arm = manifest.training.qlora.get("arm")
        if manifest_arm is not None and manifest_arm != arm:
            raise ValueError(f"CLI arm {arm} != manifest arm {manifest_arm}")

    set_seed(manifest.training.seed)
    teacher_responses = load_teacher_response_map(teacher_responses_path)
    if arm == "sequence_kd" and not teacher_responses:
        raise ValueError("sequence_kd requires pre-materialized teacher responses file")

    examples = materialize_examples(
        dataset_dir=dataset_dir,
        arm=arm,
        teacher_responses=teacher_responses,
    )
    # Deterministic order by sealed seed (same across arms before batching).
    examples = sorted(examples, key=lambda ex: ex.example_id)
    rng = random.Random(manifest.training.seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    examples = [examples[i] for i in order]

    tokenizer = load_tokenizer(
        manifest.models.student.id,
        manifest.models.student.revision,
        models_dir,
    )
    # Same-tokenizer evidence for KD arms.
    if arm in {"logit_kd", "ce_ablation"}:
        if (
            manifest.models.student.tokenizer_sha256
            != manifest.models.teacher.tokenizer_sha256
        ):
            raise RuntimeError("tokenizer sha mismatch; refusing KD/ablation")

    load_teacher_flag = arm == "logit_kd"
    student, precision_meta = load_student(
        model_id=manifest.models.student.id,
        revision=manifest.models.student.revision,
        models_dir=models_dir,
        qlora_dict=dict(manifest.training.qlora),
        max_length=manifest.training.max_length,
        load_teacher=load_teacher_flag,
    )
    teacher = None
    teacher_guard = None
    if load_teacher_flag:
        teacher, teacher_guard = load_teacher(
            model_id=manifest.models.teacher.id,
            revision=manifest.models.teacher.revision,
            models_dir=models_dir,
        )

    lr = float(manifest.training.qlora.get("learning_rate", 2e-4))
    if teacher is not None:
        optimizer = construct_optimizer_after_teacher_guard(
            lambda params: torch.optim.AdamW(params, lr=lr),
            (p for p in student.parameters() if p.requires_grad),
            teacher=teacher,
            teacher_guard=teacher_guard,
        )
    else:
        optimizer = torch.optim.AdamW(
            (p for p in student.parameters() if p.requires_grad),
            lr=lr,
        )

    temperature = float(manifest.training.qlora.get("logit_temperature", 2.0))
    kd_weight = float(manifest.training.qlora.get("kd_weight", 0.7))
    hard_ce_weight = float(manifest.training.qlora.get("hard_ce_weight", 0.3))
    vocab_chunk = int(manifest.training.qlora.get("vocab_chunk", 4096))
    max_steps = manifest.training.max_steps

    metrics_path = output_dir / "training" / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    cursor = 0
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        while step < max_steps:
            example = examples[cursor % len(examples)]
            cursor += 1
            batch = tokenize_pair(
                tokenizer,
                example.prompt_text,
                example.target_text,
                max_length=manifest.training.max_length,
            )
            device = next(student.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            student_out = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            teacher_logits = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_out = teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    teacher_logits = teacher_out.logits
            loss, parts = compute_torch_loss(
                arm=arm,
                student_logits=student_out.logits,
                labels=batch["labels"],
                completion_mask=batch["completion_mask"],
                teacher_logits=teacher_logits,
                temperature=temperature,
                kd_weight=kd_weight,
                hard_ce_weight=hard_ce_weight,
                vocab_chunk=vocab_chunk,
            )
            loss.backward()
            optimizer.step()
            step += 1
            record = {
                "step": step,
                "example_id": example.example_id,
                "arm": arm,
                **parts,
            }
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")

    adapter_dir = output_dir / "model" / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # SageMaker model channel convenience copy.
    SM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(student, PeftModel):
        student.save_pretrained(SM_MODEL_DIR)

    final_adapter = output_dir / "training" / "final"
    final_adapter.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(final_adapter)

    manifest_out = output_dir / "manifest.json"
    manifest_out.write_text(
        json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    emergency_run = {
        "arm": arm,
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.seal_sha256(),
        "max_steps": max_steps,
        "seed": manifest.training.seed,
        "student_revision": manifest.models.student.revision,
        "precision": precision_meta,
        "fake_kd": False,
        "completed_steps": step,
    }
    emergency_path = output_dir / "training" / "emergency_run.json"
    emergency_path.write_text(
        json.dumps(emergency_run, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    write_emergency_integrity(output_dir)
    return emergency_run


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="experiments.aws_smoke.train")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--arm", type=str, required=True, choices=[
        "oracle_sft",
        "ce_ablation",
        "logit_kd",
        "sequence_kd",
    ])
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--teacher-responses", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = stdout or sys.stdout
    manifest_channel = _resolve_path(None, SM_CHANNEL_MANIFEST)
    dataset_dir = _resolve_path(args.dataset_dir, SM_CHANNEL_DATASET)
    models_dir = _resolve_path(args.models_dir, SM_CHANNEL_MODELS)
    default_output = Path(os.environ.get("SM_OUTPUT_DATA_DIR", str(SM_OUTPUT_DIR)))
    output_dir = _resolve_path(args.output_dir, default_output)
    if args.manifest is not None:
        manifest_path = args.manifest
    else:
        manifest_path = discover_manifest(manifest_channel)
    try:
        manifest = load_manifest(manifest_path)
        result = run_training(
            manifest=manifest,
            arm=args.arm,  # type: ignore[arg-type]
            dataset_dir=dataset_dir,
            models_dir=models_dir,
            output_dir=output_dir,
            teacher_responses_path=args.teacher_responses,
        )
    except Exception as exc:  # noqa: BLE001 - container entrypoint must surface failures
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=out)
        return 1
    print(json.dumps({"ok": True, **result}, sort_keys=True), file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
