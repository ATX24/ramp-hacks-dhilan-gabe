#!/usr/bin/env python3
"""Measure a real base/adapter pair on sealed validation examples only."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from distillery.proof.metrics import PredictionRecord, compute_arm_metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_count(path: Path) -> int:
    count = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            count += handle.get_tensor(key).numel()
    return count


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(payload)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def render_training_prompt(example: dict[str, Any]) -> str:
    """Match the emergency trainer's sealed validation prompt exactly."""
    return json.dumps(
        {
            "task": example.get("task"),
            "difficulty": example.get("difficulty"),
            "input": example.get("input"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def generate_arm(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    arm_id: str,
    device: torch.device,
    max_new_tokens: int,
    predictions_path: Path,
) -> tuple[list[PredictionRecord], list[float]]:
    records: list[PredictionRecord] = []
    latencies: list[float] = []
    model.eval()
    with predictions_path.open("w", encoding="utf-8") as output:
        for index, example in enumerate(examples, start=1):
            prompt_text = render_training_prompt(example)
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=True,
                add_generation_prompt=True,
            )
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000.0
            new_tokens = generated[0, len(prompt_ids) :].detach().cpu().tolist()
            raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            record = PredictionRecord(
                example_id=str(example["example_id"]),
                world_id=str(example["world_id"]),
                group_id=str(example.get("group_id", "")),
                task=str(example["task"]),
                difficulty=str(example["difficulty"]),
                split="validation",
                template_family=str(example["provenance"]["template_family"]),
                arm_id=arm_id,
                seed=17,
                raw_text=raw_text,
                raw_text_provenance="captured_model_output",
                latency_ms=latency_ms,
                output_tokens=len(new_tokens),
                expected_output=dict(example["expected_output"]),
                slices={},
            )
            records.append(record)
            latencies.append(latency_ms)
            output.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
            output.flush()
            print(
                json.dumps(
                    {
                        "arm_id": arm_id,
                        "completion_tokens": len(new_tokens),
                        "event": "validation_prediction",
                        "example_id": example["example_id"],
                        "index": index,
                        "latency_ms": latency_ms,
                        "total": len(examples),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return records, latencies


def metric_payload(
    arm_id: str,
    records: list[PredictionRecord],
    latencies: list[float],
) -> dict[str, Any]:
    metrics = compute_arm_metrics(arm_id, records)
    sorted_latencies = sorted(latencies)
    p95_index = max(0, min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95)))
    return {
        "arm_id": arm_id,
        "n": metrics.n,
        "primary_index": metrics.primary_index,
        "json_parse_rate": metrics.json_parse_rate,
        "json_schema_validity": metrics.json_schema_validity,
        "transaction_joint_exact": metrics.transaction_joint_exact,
        "variance_joint_exact": metrics.variance_joint_exact,
        "cash_joint_exact": metrics.cash_joint_exact,
        "critical_invariant_violations": metrics.critical_invariant_violations,
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": statistics.median(latencies),
            "p95_nearest_rank": sorted_latencies[p95_index],
            "samples": len(latencies),
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--base-dir", type=Path, required=True)
    result.add_argument("--adapter-dir", type=Path, required=True)
    result.add_argument("--validation", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--max-examples", type=int, default=16)
    result.add_argument("--max-new-tokens", type=int, default=128)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.max_examples < 1:
        raise ValueError("--max-examples must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_jsonl(args.validation)[: args.max_examples]
    base_weights = args.base_dir / "model.safetensors"
    adapter_weights = args.adapter_dir / "adapter_model.safetensors"
    for path in (base_weights, adapter_weights):
        if not path.is_file():
            raise FileNotFoundError(path)

    device, dtype = device_and_dtype()
    torch.manual_seed(17)
    print(
        json.dumps(
            {
                "device": str(device),
                "dtype": str(dtype),
                "event": "evaluation_start",
                "examples": len(examples),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.base_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    base_records, base_latencies = generate_arm(
        model=base_model,
        tokenizer=tokenizer,
        examples=examples,
        arm_id="student_base",
        device=device,
        max_new_tokens=args.max_new_tokens,
        predictions_path=args.output_dir / "student_base.predictions.jsonl",
    )
    adapter_model = PeftModel.from_pretrained(
        base_model,
        str(args.adapter_dir),
        local_files_only=True,
    ).to(device)
    adapter_records, adapter_latencies = generate_arm(
        model=adapter_model,
        tokenizer=tokenizer,
        examples=examples,
        arm_id="oracle_sft",
        device=device,
        max_new_tokens=args.max_new_tokens,
        predictions_path=args.output_dir / "oracle_sft.predictions.jsonl",
    )

    base_metrics = metric_payload("student_base", base_records, base_latencies)
    adapter_metrics = metric_payload(
        "oracle_sft",
        adapter_records,
        adapter_latencies,
    )
    report = {
        "schema_version": "distillery.real_baseline_comparison.v1",
        "evaluation_scope": "sealed_validation_only",
        "validation_sha256": sha256_file(args.validation),
        "validation_examples": len(examples),
        "seed": 17,
        "generation": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "device": str(device),
        "dtype": str(dtype),
        "base": {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "weights_sha256": sha256_file(base_weights),
            "parameter_count": parameter_count(base_weights),
            "metrics": base_metrics,
        },
        "adapter": {
            "model_id": "model_oracle_sft",
            "artifact_id": "artifact_oracle_sft_0021b7d6cdfd",
            "weights_sha256": sha256_file(adapter_weights),
            "parameter_count": parameter_count(adapter_weights),
            "metrics": adapter_metrics,
        },
        "comparison": {
            "primary_index_delta": (
                adapter_metrics["primary_index"] - base_metrics["primary_index"]
            ),
            "improvement_claimed": False,
            "proof_status": "insufficient_evidence",
            "limitations": [
                "emergency smoke validation has no transaction_review examples",
                "single seed",
                "local hardware differs from SageMaker training hardware",
                "no bootstrap confidence interval",
            ],
        },
    }
    report_path = args.output_dir / "comparison.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "evaluation_complete", **report["comparison"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
