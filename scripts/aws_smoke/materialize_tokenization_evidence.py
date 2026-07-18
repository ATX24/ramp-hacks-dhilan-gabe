#!/usr/bin/env python3
"""Create completion-count evidence from exact offline tokenizer snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from distillery.contracts.hashing import content_sha256, sha256_hex  # noqa: E402
from experiments.aws_smoke.model_evidence import (  # noqa: E402
    assert_loaded_tokenizers_compatible,
    require_local_snapshot,
    verify_model_config_sha256,
    verify_tokenizer_runtime_evidence,
)
from experiments.aws_smoke.pins import load_evidence  # noqa: E402
from experiments.aws_smoke.profile import DEFAULT_EMERGENCY_PROFILE  # noqa: E402
from experiments.aws_smoke.tokenization import (  # noqa: E402
    ArmTokenizationEvidence,
    TokenizationEvidence,
    build_chat_token_pair,
    canonical_completion_records_sha256,
    completion_record_sha256,
    teacher_responses_sha256,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object in {path}")
            rows.append(payload)
    return rows


def _prompt(row: dict) -> str:
    return json.dumps(
        {
            "task": row.get("task"),
            "difficulty": row.get("difficulty"),
            "input": row.get("input"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _oracle_target(row: dict) -> str:
    target = row.get("expected_output")
    if not isinstance(target, dict) or not target:
        raise ValueError(f"example {row.get('example_id')} lacks oracle output")
    return json.dumps(target, sort_keys=True, ensure_ascii=False)


def _response_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in _load_jsonl(path):
        example_id = str(row["example_id"])
        response = row.get("response_text")
        if not isinstance(response, str) or not response:
            raise ValueError(f"empty teacher response for {example_id}")
        result[example_id] = response
    return result


def _arm_evidence(
    *,
    arm: str,
    rows: list[dict],
    tokenizer: object,
    targets: dict[str, str],
    teacher_sha256: str | None,
    source_file_sha256: str,
) -> ArmTokenizationEvidence:
    profile = DEFAULT_EMERGENCY_PROFILE
    counts: dict[str, int] = {}
    prompt_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    record_hashes: dict[str, str] = {}
    completion_hashes: dict[str, str] = {}
    originals: dict[str, int] = {}
    truncated: list[str] = []
    for row in rows:
        example_id = str(row["example_id"])
        pair = build_chat_token_pair(
            tokenizer,
            _prompt(row),
            targets[example_id],
            max_length=profile.max_length,
            max_completion=profile.max_completion,
        )
        counts[example_id] = pair.completion_token_count
        prompt_counts[example_id] = pair.prompt_token_count
        total_counts[example_id] = len(pair.input_ids)
        record_hashes[example_id] = content_sha256(row)
        completion_hashes[example_id] = completion_record_sha256(
            example_id=example_id,
            target_text=targets[example_id],
            target_source=(
                "pre_materialized_teacher" if arm == "sequence_kd" else "oracle"
            ),
        )
        originals[example_id] = pair.original_completion_token_count
        if pair.completion_truncated:
            truncated.append(example_id)
    return ArmTokenizationEvidence(
        arm=arm,
        target_source=(
            "pre_materialized_teacher" if arm == "sequence_kd" else "oracle"
        ),
        completion_token_counts=counts,
        prompt_token_counts=prompt_counts,
        total_token_counts=total_counts,
        record_sha256=record_hashes,
        source_file_sha256=source_file_sha256,
        canonical_records_sha256=canonical_completion_records_sha256(
            completion_hashes
        ),
        completion_record_sha256=completion_hashes,
        original_completion_token_counts=originals,
        truncated_example_ids=tuple(sorted(truncated)),
        teacher_responses_sha256=teacher_sha256,
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="materialize_tokenization_evidence")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-responses", type=Path, default=None)
    args = parser.parse_args()

    evidence = load_evidence(args.evidence)
    student_snapshot = require_local_snapshot(
        args.models_dir,
        evidence.student_model_id,
        evidence.student_revision,
    )
    teacher_snapshot = require_local_snapshot(
        args.models_dir,
        evidence.teacher_model_id,
        evidence.teacher_revision,
    )
    verify_model_config_sha256(
        student_snapshot,
        evidence.student_model_config_sha256,
    )
    verify_model_config_sha256(
        teacher_snapshot,
        evidence.teacher_model_config_sha256,
    )
    student_tokenizer = AutoTokenizer.from_pretrained(
        str(student_snapshot),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        str(teacher_snapshot),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    student_runtime = verify_tokenizer_runtime_evidence(
        snapshot_dir=student_snapshot,
        tokenizer=student_tokenizer,
        expected_tokenizer_sha256=evidence.student_tokenizer_sha256,
        expected_chat_template_sha256=evidence.student_chat_template_sha256,
        expected_special_token_map=evidence.student_special_token_map,
    )
    teacher_runtime = verify_tokenizer_runtime_evidence(
        snapshot_dir=teacher_snapshot,
        tokenizer=teacher_tokenizer,
        expected_tokenizer_sha256=evidence.teacher_tokenizer_sha256,
        expected_chat_template_sha256=evidence.teacher_chat_template_sha256,
        expected_special_token_map=evidence.teacher_special_token_map,
    )
    assert_loaded_tokenizers_compatible(teacher_runtime, student_runtime)

    train_path = args.subset_dir / "train.jsonl"
    rows = _load_jsonl(train_path)
    train_file_sha256 = sha256_hex(train_path.read_bytes())
    oracle_targets = {str(row["example_id"]): _oracle_target(row) for row in rows}
    oracle = _arm_evidence(
        arm="oracle_sft",
        rows=rows,
        tokenizer=student_tokenizer,
        targets=oracle_targets,
        teacher_sha256=None,
        source_file_sha256=train_file_sha256,
    )
    arms = {
        "oracle_sft": oracle,
        "ce_ablation": oracle.model_copy(update={"arm": "ce_ablation"}),
        "logit_kd": oracle.model_copy(update={"arm": "logit_kd"}),
    }
    if args.teacher_responses is not None:
        response_sha256 = teacher_responses_sha256(args.teacher_responses)
        responses = _response_map(args.teacher_responses)
        missing = sorted(set(oracle_targets) - set(responses))
        if missing:
            raise ValueError(
                f"teacher responses do not cover all training examples: {missing}"
            )
        arms["sequence_kd"] = _arm_evidence(
            arm="sequence_kd",
            rows=rows,
            tokenizer=student_tokenizer,
            targets=responses,
            teacher_sha256=response_sha256,
            source_file_sha256=response_sha256,
        )
    profile = DEFAULT_EMERGENCY_PROFILE
    payload = TokenizationEvidence(
        student_tokenizer_sha256=student_runtime.tokenizer_sha256,
        student_chat_template_sha256=student_runtime.chat_template_sha256,
        student_special_token_map=student_runtime.special_token_map,
        max_length=profile.max_length,
        max_completion=profile.max_completion,
        arms=arms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "arms": sorted(arms),
                "offline_only": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
