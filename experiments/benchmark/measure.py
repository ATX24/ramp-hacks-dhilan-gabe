"""Transformers-only decode throughput measurement (no fabricated metrics)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.benchmark import ARTIFACT_LABEL, RUNTIME_LABEL, SCHEMA_VERSION
from experiments.benchmark.prompts import BenchmarkPrompt, build_benchmark_prompts
from experiments.benchmark.stats import percentile

STUDENT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TEACHER_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
STUDENT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TEACHER_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    revision: str
    role: str
    local_path: Path


def _nvidia_smi_snapshot() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}
    rows = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "name": parts[0],
                "utilization_gpu_pct": float(parts[1]),
                "utilization_memory_pct": float(parts[2]),
                "memory_used_mib": float(parts[3]),
                "memory_total_mib": float(parts[4]),
            }
        )
    return {"available": True, "gpus": rows}


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name!r}")
    return mapping[name]


def discover_model_specs(models_root: Path) -> list[ModelSpec]:
    """Locate sealed student/teacher snapshots under a SageMaker models channel."""
    specs: list[ModelSpec] = []
    candidates = [
        (
            STUDENT_MODEL_ID,
            STUDENT_REVISION,
            "student",
            models_root / "Qwen" / "Qwen2.5-0.5B-Instruct" / STUDENT_REVISION,
        ),
        (
            TEACHER_MODEL_ID,
            TEACHER_REVISION,
            "teacher",
            models_root / "Qwen" / "Qwen2.5-1.5B-Instruct" / TEACHER_REVISION,
        ),
    ]
    # SageMaker may flatten channel prefixes; also accept direct revision dirs.
    for model_id, revision, role, preferred in candidates:
        path = preferred
        if not path.is_dir():
            alt = models_root / revision
            if alt.is_dir():
                path = alt
            else:
                # Walk one level for a directory containing config.json + weights.
                found: Path | None = None
                for candidate in models_root.rglob("config.json"):
                    parent = candidate.parent
                    if (parent / "model.safetensors").is_file() or any(
                        parent.glob("*.safetensors")
                    ):
                        # Prefer matching revision directory name.
                        if parent.name == revision or revision in str(parent):
                            found = parent
                            break
                        if found is None:
                            found = parent
                if found is None:
                    raise FileNotFoundError(
                        f"missing local snapshot for {model_id}@{revision} under {models_root}"
                    )
                path = found
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"snapshot missing config.json: {path}")
        specs.append(
            ModelSpec(
                model_id=model_id,
                revision=revision,
                role=role,
                local_path=path,
            )
        )
    return specs


def _encode_prompts(
    tokenizer: Any,
    prompts: Sequence[BenchmarkPrompt],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    texts: list[str] = []
    for prompt in prompts:
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                list(prompt.messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt.prompt_text
        texts.append(text)
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_lens = [int(x) for x in attention_mask.sum(dim=1).tolist()]
    return input_ids, attention_mask, prompt_lens


def _generate_batch(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[BenchmarkPrompt],
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    input_ids, attention_mask, prompt_lens = _encode_prompts(
        tokenizer, prompts, device=device
    )
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    per_row_completion = [
        max(0, int(row.shape[-1]) - prompt_len)
        for row, prompt_len in zip(output, prompt_lens, strict=True)
    ]
    return {
        "elapsed_s": elapsed,
        "prompt_tokens": int(sum(prompt_lens)),
        "completion_tokens": int(sum(per_row_completion)),
        "completion_tokens_per_row": per_row_completion,
        "batch_size": len(prompts),
        "latency_ms": elapsed * 1000.0,
    }


def measure_model(
    spec: ModelSpec,
    *,
    dtype_name: str,
    max_new_tokens: int,
    warmups: int,
    timed: int,
    batch_sizes: Sequence[int],
    seed: int,
    hardware: str,
    instance_type: str,
) -> dict[str, Any]:
    warm_prompts, timed_prompts = build_benchmark_prompts(warmups=warmups, timed=timed, seed=seed)
    if any(b < 1 for b in batch_sizes):
        raise ValueError("batch sizes must be positive")

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = resolve_dtype(dtype_name)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        str(spec.local_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    cold_load_s = time.perf_counter() - load_started

    # Cold first request: model already loaded, first generate after load.
    cold_first = _generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=warm_prompts[:1] or timed_prompts[:1],
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    cold_startup_s = cold_load_s + float(cold_first["elapsed_s"])

    # Warmups (not timed).
    for idx in range(0, len(warm_prompts), 1):
        _generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=warm_prompts[idx : idx + 1],
            max_new_tokens=max_new_tokens,
            seed=seed + idx,
        )

    warm_first = _generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=timed_prompts[:1],
        max_new_tokens=max_new_tokens,
        seed=seed + 10_000,
    )

    profiles: list[dict[str, Any]] = []
    peak_alloc = 0
    peak_reserved = 0
    smi_samples: list[dict[str, Any]] = []

    for batch_size in batch_sizes:
        latencies_ms: list[float] = []
        decode_tps_samples: list[float] = []
        total_completion = 0
        total_prompt = 0
        failures = 0
        wall_started = time.perf_counter()
        # For batch>1, consume timed prompts in non-overlapping windows.
        cursor = 0
        n_steps = math.ceil(len(timed_prompts) / batch_size)
        for step in range(n_steps):
            window = timed_prompts[cursor : cursor + batch_size]
            cursor += batch_size
            if len(window) < batch_size:
                # Repeat-fill from the front to keep a full batch without new labels.
                need = batch_size - len(window)
                window = list(window) + list(timed_prompts[:need])
            try:
                result = _generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=window,
                    max_new_tokens=max_new_tokens,
                    seed=seed + 20_000 + step,
                )
            except Exception:
                failures += 1
                continue
            latencies_ms.append(float(result["latency_ms"]))
            total_completion += int(result["completion_tokens"])
            total_prompt += int(result["prompt_tokens"])
            if batch_size == 1 and float(result["elapsed_s"]) > 0:
                decode_tps_samples.append(
                    float(result["completion_tokens"]) / float(result["elapsed_s"])
                )
            if device.type == "cuda":
                peak_alloc = max(peak_alloc, int(torch.cuda.max_memory_allocated()))
                peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved()))
            if step % max(1, n_steps // 5) == 0:
                smi_samples.append(_nvidia_smi_snapshot())
        wall_s = time.perf_counter() - wall_started
        successful_requests = max(0, (n_steps - failures) * batch_size)
        # Aggregate output tok/s over the timed window wall clock.
        output_tps = (total_completion / wall_s) if wall_s > 0 else 0.0
        decode_tps = (
            sum(decode_tps_samples) / len(decode_tps_samples)
            if decode_tps_samples
            else output_tps
        )

        timed_by_task = Counter(p.task for p in timed_prompts)
        warmup_by_task = Counter(p.task for p in warm_prompts)
        profiles.append(
            {
                "hardware": hardware,
                "instance_type": instance_type,
                "runtime": RUNTIME_LABEL,
                "runtime_family": "transformers",
                "vllm_tested": False,
                "artifact_label": ARTIFACT_LABEL,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "role": spec.role,
                "dtype": dtype_name,
                "max_new_tokens": max_new_tokens,
                "batch_size": batch_size,
                "warmup_requests": len(warm_prompts),
                "timed_examples": len(timed_prompts),
                "warmup_requests_by_task": dict(warmup_by_task),
                "timed_examples_by_task": dict(timed_by_task),
                "latency_p50_ms": percentile(latencies_ms, 0.50),
                "latency_p95_ms": percentile(latencies_ms, 0.95),
                "latency_samples_ms": latencies_ms,
                "requests_per_second": (successful_requests / wall_s) if wall_s > 0 else 0.0,
                "output_tokens_per_second": output_tps,
                "single_request_decode_tokens_per_second": decode_tps,
                "prompt_tokens_total": total_prompt,
                "completion_tokens_total": total_completion,
                "mean_prompt_tokens": (
                    total_prompt / max(1, len(latencies_ms) * batch_size)
                ),
                "mean_completion_tokens": (
                    total_completion / max(1, len(latencies_ms) * batch_size)
                ),
                "failure_rate": failures / max(1, n_steps),
                "failures": failures,
                "wall_time_seconds": wall_s,
                "peak_vram_allocated_gb": peak_alloc / (1024**3),
                "peak_vram_reserved_gb": peak_reserved / (1024**3),
                "gpu_snapshots": smi_samples,
                "notes": [
                    "base_model_proxy_until_trained_adapters",
                    "transformers_runtime_only",
                    "train_validation_prompts_only_no_test_split",
                ],
            }
        )

    del model
    torch.cuda.empty_cache()
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "role": spec.role,
        "artifact_label": ARTIFACT_LABEL,
        "runtime": RUNTIME_LABEL,
        "runtime_family": "transformers",
        "vllm_tested": False,
        "dtype": dtype_name,
        "hardware": hardware,
        "instance_type": instance_type,
        "cold_startup": {
            "model_load_seconds": cold_load_s,
            "first_request_e2e_seconds": cold_startup_s,
            "first_request_generate_seconds": float(cold_first["elapsed_s"]),
            "first_request_completion_tokens": int(cold_first["completion_tokens"]),
            "first_request_prompt_tokens": int(cold_first["prompt_tokens"]),
        },
        "warm_startup": {
            "first_request_generate_seconds": float(warm_first["elapsed_s"]),
            "first_request_e2e_latency_ms": float(warm_first["latency_ms"]),
            "first_request_completion_tokens": int(warm_first["completion_tokens"]),
            "first_request_prompt_tokens": int(warm_first["prompt_tokens"]),
        },
        "profiles": profiles,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_benchmark_suite(
    *,
    models_root: Path,
    output_dir: Path,
    dtype_name: str,
    max_new_tokens: int,
    warmups: int,
    timed: int,
    batch_sizes: Sequence[int],
    seed: int,
    hardware: str,
    instance_type: str,
) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    started = time.perf_counter()
    specs = discover_model_specs(models_root)
    results = []
    for spec in specs:
        results.append(
            measure_model(
                spec,
                dtype_name=dtype_name,
                max_new_tokens=max_new_tokens,
                warmups=warmups,
                timed=timed,
                batch_sizes=batch_sizes,
                seed=seed,
                hardware=hardware,
                instance_type=instance_type,
            )
        )
    suite = {
        "schema_version": SCHEMA_VERSION,
        "runtime": RUNTIME_LABEL,
        "runtime_family": "transformers",
        "vllm_tested": False,
        "artifact_label": ARTIFACT_LABEL,
        "hardware": hardware,
        "instance_type": instance_type,
        "dtype": dtype_name,
        "max_new_tokens": max_new_tokens,
        "warmups": warmups,
        "timed": timed,
        "batch_sizes": list(batch_sizes),
        "seed": seed,
        "gpu_snapshot_start": _nvidia_smi_snapshot(),
        "models": results,
        "wall_time_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "benchmark_suite.json", suite)
    for model_result in results:
        safe = model_result["model_id"].replace("/", "__")
        write_json(output_dir / f"profile_{safe}.json", model_result)
        for profile in model_result["profiles"]:
            write_json(
                output_dir
                / f"systems_{safe}_batch{profile['batch_size']}.json",
                profile,
            )
    return suite
