"""Deterministic sealed teacher-response synthesis for smoke train/validation.

Generates greedy teacher completions for finance_world.v1 smoke splits only.
Never loads test / IID / OOD examples into the teacher context, and never
supplies expected_output / oracle fields to the model.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from distillery.contracts.hashing import content_sha256, sha256_hex
from distillery.contracts.tasks import LabelSource, SplitName, TaskId
from distillery.data.generate import CORPUS_SMOKE, generate_corpus
from distillery.data.validate import validate_output
from distillery.recipes.base import ResponseRecord
from distillery.recipes.sequence_v1 import (
    materialize_sequence_examples,
    retokenize_text_pair,
    validate_response_text,
)
from experiments.aws_smoke.model_evidence import (
    require_local_model_weights,
    require_local_snapshot,
    verify_model_config_sha256,
)
from experiments.aws_smoke.profile import DEFAULT_EMERGENCY_PROFILE


def render_prompt(example: Mapping[str, Any]) -> str:
    """Match emergency trainer prompt rendering without importing train.py."""
    return json.dumps(
        {
            "task": example.get("task"),
            "difficulty": example.get("difficulty"),
            "input": example.get("input"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )

SCHEMA_VERSION = "distillery.teacher_synthesis.materialization.v1"
TEACHER_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TEACHER_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
EXPECTED_MODEL_MATERIALIZATION_SHA256 = (
    "7e6d59fa8d30805e168a52bc4ef4225e2dff900159489c71cf62e5dde9fd70e6"
)
EXPECTED_TEACHER_WEIGHT_SHA256 = (
    "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
)
EXPECTED_TEACHER_CONFIG_SHA256 = (
    "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670"
)
MODEL_MATERIALIZATION_URI = (
    "s3://distillery-225989358036-us-east-1/models/materialization.json"
)
DISTILLERY_BUCKET = "distillery-225989358036-us-east-1"
FORBIDDEN_TEACHER_KEYS = frozenset(
    {
        "answer",
        "expected_output",
        "label",
        "oracle",
        "predicted_output",
        "target",
        "target_output",
    }
)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

SplitLiteral = Literal["train", "validation"]
SYNTHESIS_SPLITS: tuple[SplitLiteral, ...] = ("train", "validation")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    do_sample: Literal[False] = False
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = DEFAULT_EMERGENCY_PROFILE.max_completion
    seed: int = DEFAULT_EMERGENCY_PROFILE.seed

    def as_dict(self) -> dict[str, Any]:
        return {
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "decoding": "greedy_temperature_zero",
        }


@dataclass(frozen=True, slots=True)
class SealedTeacherRow:
    example_id: str
    split: SplitLiteral
    task: str
    difficulty: str
    prompt_text: str
    prompt_sha256: str
    raw_response_text: str
    raw_response_sha256: str
    canonical_response_text: str
    canonical_response_sha256: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    prompt_token_count: int
    completion_token_count: int
    total_token_count: int
    record: ResponseRecord
    teacher_payload: dict[str, Any]


def verify_model_materialization_bytes(payload: bytes) -> dict[str, Any]:
    digest = sha256_hex(payload)
    if digest != EXPECTED_MODEL_MATERIALIZATION_SHA256:
        raise ValueError(
            "model materialization hash mismatch: "
            f"expected={EXPECTED_MODEL_MATERIALIZATION_SHA256} actual={digest}"
        )
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("model materialization must be a JSON object")
    if data.get("schema") != "distillery.models.materialization.v1":
        raise ValueError("unexpected model materialization schema")
    teacher = _teacher_entry(data)
    weight = teacher["files"]["model.safetensors"]["sha256"]
    if weight != EXPECTED_TEACHER_WEIGHT_SHA256:
        raise ValueError(
            "teacher weight hash mismatch against sealed materialization: "
            f"expected={EXPECTED_TEACHER_WEIGHT_SHA256} actual={weight}"
        )
    config_sha = teacher["files"]["config.json"]["sha256"]
    if config_sha != EXPECTED_TEACHER_CONFIG_SHA256:
        raise ValueError(
            "teacher config hash mismatch against sealed materialization: "
            f"expected={EXPECTED_TEACHER_CONFIG_SHA256} actual={config_sha}"
        )
    if teacher.get("revision") != TEACHER_REVISION:
        raise ValueError(
            f"teacher revision mismatch: expected={TEACHER_REVISION} "
            f"actual={teacher.get('revision')}"
        )
    return data


def _teacher_entry(materialization: Mapping[str, Any]) -> dict[str, Any]:
    for model in materialization.get("models", []):
        if not isinstance(model, dict):
            continue
        model_id = model.get("model_id") or model.get("id")
        if model_id == TEACHER_MODEL_ID and model.get("revision") == TEACHER_REVISION:
            return model
    raise ValueError("pinned teacher model missing from materialization manifest")


def assert_teacher_safe_prompt_payload(payload: Mapping[str, Any]) -> None:
    """Fail loud if any forbidden label/oracle field is present for the teacher."""
    stack: list[Any] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, value in current.items():
                if key in FORBIDDEN_TEACHER_KEYS:
                    raise ValueError(
                        f"refusing to expose forbidden teacher field {key!r}"
                    )
                stack.append(value)
        elif isinstance(current, (list, tuple)):
            stack.extend(current)


def extract_json_object(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(stripped)
        if match is None:
            return None
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
    else:
        candidate = stripped
    if not isinstance(parsed, dict):
        return None
    return candidate


def canonicalize_response_json(raw_object_text: str) -> str:
    parsed = json.loads(raw_object_text)
    if not isinstance(parsed, dict):
        raise ValueError("canonical response must be a JSON object")
    return json.dumps(parsed, sort_keys=True, ensure_ascii=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_teacher_snapshot_from_materialization(
    *,
    materialization: Mapping[str, Any],
    models_dir: Path,
    download_file: Callable[[str, Path], None],
) -> Path:
    """Materialize the pinned teacher snapshot under models/<org>/<name>/<rev>/."""
    teacher = _teacher_entry(materialization)
    org, _, name = TEACHER_MODEL_ID.partition("/")
    dest = models_dir / org / name / TEACHER_REVISION
    dest.mkdir(parents=True, exist_ok=True)
    for filename, meta in sorted(teacher["files"].items()):
        if filename in {"SHA256SUMS", "snapshot-manifest.json"}:
            continue
        expected = str(meta["sha256"])
        target = dest / filename
        if target.is_file() and file_sha256(target) == expected:
            continue
        uri = str(meta["s3_uri"])
        download_file(uri, target)
        actual = file_sha256(target)
        if actual != expected:
            target.unlink(missing_ok=True)
            raise ValueError(
                f"downloaded {filename} hash mismatch: "
                f"expected={expected} actual={actual}"
            )
    require_local_model_weights(dest)
    verify_model_config_sha256(dest, EXPECTED_TEACHER_CONFIG_SHA256)
    weight_path = dest / "model.safetensors"
    if file_sha256(weight_path) != EXPECTED_TEACHER_WEIGHT_SHA256:
        raise ValueError("local teacher weight hash diverged after sync")
    return require_local_snapshot(models_dir, TEACHER_MODEL_ID, TEACHER_REVISION)


def load_smoke_train_validation_rows() -> dict[SplitLiteral, list[dict[str, Any]]]:
    """Generate sealed smoke corpus and return only train/validation rows."""
    corpus = generate_corpus(CORPUS_SMOKE, validate=True, check_near_duplicates=True)
    out: dict[SplitLiteral, list[dict[str, Any]]] = {}
    for split in SYNTHESIS_SPLITS:
        rows = [
            example.model_dump(mode="json")
            for example in corpus.by_split[SplitName(split)]
        ]
        if not rows:
            raise ValueError(f"smoke {split} split is empty")
        # Defense in depth: synthesis never carries held-out splits.
        for row in rows:
            prov_split = row.get("provenance", {}).get("split")
            if prov_split != split:
                raise ValueError(
                    f"unexpected provenance.split={prov_split!r} in {split}"
                )
        out[split] = rows
    held_out_ids = {
        example.example_id
        for split_name, examples in corpus.by_split.items()
        if split_name.value not in SYNTHESIS_SPLITS
        for example in examples
    }
    synthesis_ids = {row["example_id"] for rows in out.values() for row in rows}
    overlap = sorted(held_out_ids & synthesis_ids)
    if overlap:
        raise ValueError(f"held-out example ids leaked into synthesis: {overlap[:5]}")
    return out


def build_teacher_prompt(row: Mapping[str, Any]) -> str:
    payload = {
        "task": row.get("task"),
        "difficulty": row.get("difficulty"),
        "input": row.get("input"),
    }
    assert_teacher_safe_prompt_payload(payload)
    # Match emergency trainer prompt rendering exactly.
    prompt = render_prompt(
        {
            "task": row["task"],
            "difficulty": row["difficulty"],
            "input": row["input"],
        }
    )
    assert_teacher_safe_prompt_payload(json.loads(prompt))
    return prompt


def _encode_joint_with_offsets(
    tokenizer: Any,
    text: str,
) -> dict[str, list]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    if not input_ids:
        raise ValueError("tokenizer produced empty joint encoding")
    return {"input_ids": input_ids, "offset_mapping": offsets}


def evaluate_teacher_response(
    *,
    task: str,
    raw_response: str,
) -> tuple[str, tuple[str, ...], bool]:
    """Return (canonical_or_raw_text, rejection_reasons, accepted)."""
    reasons = list(validate_response_text(raw_response))
    extracted = extract_json_object(raw_response)
    if extracted is None:
        if "invalid_json" not in reasons and "empty_response" not in reasons:
            reasons.append("invalid_json")
        return raw_response.strip() or raw_response, tuple(reasons), False
    try:
        canonical = canonicalize_response_json(extracted)
    except (TypeError, ValueError, json.JSONDecodeError):
        reasons.append("canonicalization_failed")
        return extracted, tuple(reasons), False
    parsed = json.loads(canonical)
    schema = validate_output(TaskId(task), parsed)
    if not schema.ok:
        reasons.extend(f"finance_invariant:{error}" for error in schema.errors)
    accepted = not reasons
    return canonical, tuple(reasons), accepted


def seal_teacher_row(
    *,
    row: Mapping[str, Any],
    split: SplitLiteral,
    prompt_text: str,
    raw_response: str,
    tokenizer: Any,
    tokenizer_sha256: str,
    chat_template_sha256_value: str,
    teacher_weight_sha256: str,
    generation_config: GenerationConfig,
    prompt_token_count: int,
    generation_completion_token_count: int,
) -> SealedTeacherRow:
    canonical, reasons, accepted = evaluate_teacher_response(
        task=str(row["task"]),
        raw_response=raw_response,
    )
    selected = canonical if accepted else (canonical or raw_response)
    lineage = ["teacher_generation"]
    if accepted and selected != raw_response.strip():
        lineage.append("json_canonicalize_v1")
    generation_params = {
        **generation_config.as_dict(),
        "prompt_sha256": content_sha256(prompt_text),
        "raw_response_sha256": content_sha256(raw_response),
        "canonical_response_sha256": content_sha256(selected),
        "chat_template_sha256": chat_template_sha256_value,
        "tokenizer_sha256": tokenizer_sha256,
        "teacher_weight_sha256": teacher_weight_sha256,
        "teacher_model_config_sha256": EXPECTED_TEACHER_CONFIG_SHA256,
        "model_materialization_sha256": EXPECTED_MODEL_MATERIALIZATION_SHA256,
        "generation_prompt_token_count": prompt_token_count,
        "generation_completion_token_count": generation_completion_token_count,
        "rejection_reasons": list(reasons),
        "accepted": accepted,
        "split": split,
    }
    tokenization = retokenize_text_pair(
        prompt_text,
        selected,
        tokenizer_sha256=tokenizer_sha256,
        encode_with_offsets_fn=lambda text: _encode_joint_with_offsets(tokenizer, text),
    )
    record = ResponseRecord.seal(
        example_id=str(row["example_id"]),
        task=str(row["task"]),
        difficulty=str(row["difficulty"]),
        prompt_text=prompt_text,
        response_text=raw_response,
        selected_target_text=selected,
        label_source=LabelSource.TEACHER,
        tokenization=tokenization,
        teacher_model_id=TEACHER_MODEL_ID,
        teacher_revision=TEACHER_REVISION,
        generation_params=generation_params,
        transformation_lineage=tuple(lineage),
    )
    teacher_payload = {
        "example_id": record.example_id,
        "split": split,
        "task": record.task,
        "difficulty": record.difficulty,
        "response_text": selected if accepted else raw_response,
        "raw_response_text": raw_response,
        "canonical_response_text": selected,
        "label_source": LabelSource.TEACHER.value,
        "accepted": accepted,
        "rejection_reasons": list(reasons),
        "prompt_sha256": generation_params["prompt_sha256"],
        "raw_response_sha256": generation_params["raw_response_sha256"],
        "canonical_response_sha256": generation_params["canonical_response_sha256"],
        "record_sha256": record.record_sha256,
        "prompt_token_count": record.prompt_token_count,
        "completion_token_count": record.completion_token_count,
        "total_token_count": record.total_token_count,
        "teacher_model_id": TEACHER_MODEL_ID,
        "teacher_revision": TEACHER_REVISION,
        "teacher_weight_sha256": teacher_weight_sha256,
        "generation_params": generation_params,
    }
    return SealedTeacherRow(
        example_id=record.example_id,
        split=split,
        task=record.task,
        difficulty=record.difficulty,
        prompt_text=prompt_text,
        prompt_sha256=str(generation_params["prompt_sha256"]),
        raw_response_text=raw_response,
        raw_response_sha256=str(generation_params["raw_response_sha256"]),
        canonical_response_text=selected,
        canonical_response_sha256=str(generation_params["canonical_response_sha256"]),
        accepted=accepted,
        rejection_reasons=reasons,
        prompt_token_count=record.prompt_token_count,
        completion_token_count=record.completion_token_count,
        total_token_count=record.total_token_count,
        record=record,
        teacher_payload=teacher_payload,
    )


def select_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_generation_determinism(seed: int) -> None:
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_one(
    *,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    device: str,
    generation_config: GenerationConfig,
) -> tuple[str, int, int]:
    import torch

    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(prompt_ids, "to"):
        input_ids = prompt_ids.to(device)
    else:
        input_ids = torch.tensor([list(prompt_ids)], device=device)
    prompt_token_count = int(input_ids.shape[-1])
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=generation_config.max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    completion_ids = output[0, prompt_token_count:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return text, prompt_token_count, int(completion_ids.shape[-1])


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    payload = ""
    if rows:
        payload = (
            "\n".join(
                json.dumps(dict(row), sort_keys=True, ensure_ascii=False) for row in rows
            )
            + "\n"
        )
    path.write_text(payload, encoding="utf-8")
    return sha256_hex(payload.encode("utf-8"))


def count_by_task(
    rows: Sequence[SealedTeacherRow],
    *,
    accepted_only: bool | None = None,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        if accepted_only is True and not row.accepted:
            continue
        if accepted_only is False and row.accepted:
            continue
        counter[row.task] += 1
    return dict(sorted(counter.items()))


def build_materialization_manifest(
    *,
    synthesis_id: str,
    s3_prefix: str,
    rows: Sequence[SealedTeacherRow],
    file_sha256_map: Mapping[str, str],
    tokenizer_sha256: str,
    chat_template_digest: str,
    teacher_weight_sha256: str,
    generation_config: GenerationConfig,
    device: str,
    runtime_seconds: float,
    cost_usd: float,
    corpus_content_sha256: str,
) -> dict[str, Any]:
    by_split: dict[str, list[SealedTeacherRow]] = {split: [] for split in SYNTHESIS_SPLITS}
    for row in rows:
        by_split[row.split].append(row)
    counts: dict[str, Any] = {}
    for split, split_rows in by_split.items():
        accepted = [row for row in split_rows if row.accepted]
        rejected = [row for row in split_rows if not row.accepted]
        counts[split] = {
            "total": len(split_rows),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "accepted_by_task": count_by_task(split_rows, accepted_only=True),
            "rejected_by_task": count_by_task(split_rows, accepted_only=False),
            "total_by_task": count_by_task(split_rows),
        }
    report = materialize_sequence_examples([row.record for row in rows])
    return {
        "schema_version": SCHEMA_VERSION,
        "synthesis_id": synthesis_id,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "s3_prefix": s3_prefix.rstrip("/") + "/",
        "bucket": DISTILLERY_BUCKET,
        "corpus": "smoke",
        "corpus_schema_version": "finance_world.v1",
        "splits": list(SYNTHESIS_SPLITS),
        "held_out_excluded": ["test", "iid_test", "ood_test"],
        "teacher_model_id": TEACHER_MODEL_ID,
        "teacher_revision": TEACHER_REVISION,
        "teacher_weight_sha256": teacher_weight_sha256,
        "teacher_model_config_sha256": EXPECTED_TEACHER_CONFIG_SHA256,
        "tokenizer_sha256": tokenizer_sha256,
        "chat_template_sha256": chat_template_digest,
        "model_materialization_uri": MODEL_MATERIALIZATION_URI,
        "model_materialization_sha256": EXPECTED_MODEL_MATERIALIZATION_SHA256,
        "generation_config": generation_config.as_dict(),
        "label_source": LabelSource.TEACHER.value,
        "corpus_content_sha256": corpus_content_sha256,
        "counts": counts,
        "materialization_report": {
            "accepted": len(report.accepted),
            "rejected": len(report.rejected),
            "label_source_counts": report.label_source_counts,
            "recipe_id": report.recipe_id,
        },
        "file_sha256": dict(sorted(file_sha256_map.items())),
        "launcher_files": {
            "teacher_responses": "teacher_responses.jsonl",
            "response_records": "response_records.jsonl",
            "dataset_train": "train.jsonl",
            "dataset_validation": "validation.jsonl",
            "materialization": "materialization.json",
        },
        "runtime": {
            "device": device,
            "elapsed_seconds": runtime_seconds,
            "cost_usd": cost_usd,
            "backend": "local",
        },
        "immutable": True,
    }


def unique_synthesis_prefix() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    nonce = sha256_hex(os.urandom(16))[:12]
    return (
        f"s3://{DISTILLERY_BUCKET}/synthesis/"
        f"smoke-teacher-v1-{stamp}-{nonce}"
    )
