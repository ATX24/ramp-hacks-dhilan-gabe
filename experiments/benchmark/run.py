#!/usr/bin/env python3
"""Container entrypoint for finite Transformers throughput benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.benchmark.measure import run_benchmark_suite  # noqa: E402

DEFAULT_MODELS = Path(os.environ.get("SM_CHANNEL_MODELS", "/opt/ml/input/data/models"))
DEFAULT_OUTPUT = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
FAILURE_PATH = Path("/opt/ml/output/failure")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distillery Transformers benchmark")
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--timed", type=int, default=200)
    parser.add_argument("--batch-sizes", default="1,8")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--hardware",
        default=os.environ.get("DISTILLERY_BENCHMARK_HARDWARE", "NVIDIA-A10G"),
    )
    parser.add_argument(
        "--instance-type",
        default=os.environ.get("DISTILLERY_BENCHMARK_INSTANCE_TYPE", "ml.g5.xlarge"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    batch_sizes = [int(part.strip()) for part in args.batch_sizes.split(",") if part.strip()]
    try:
        suite = run_benchmark_suite(
            models_root=args.models_root,
            output_dir=args.output_dir,
            dtype_name=args.dtype,
            max_new_tokens=args.max_new_tokens,
            warmups=args.warmups,
            timed=args.timed,
            batch_sizes=batch_sizes,
            seed=args.seed,
            hardware=args.hardware,
            instance_type=args.instance_type,
        )
        summary_path = args.output_dir / "benchmark_summary.json"
        summary = {
            "status": "succeeded",
            "instance_type": args.instance_type,
            "hardware": args.hardware,
            "dtype": args.dtype,
            "runtime": suite["runtime"],
            "models": [
                {
                    "model_id": model["model_id"],
                    "revision": model["revision"],
                    "profiles": [
                        {
                            "batch_size": profile["batch_size"],
                            "output_tokens_per_second": profile["output_tokens_per_second"],
                            "single_request_decode_tokens_per_second": profile[
                                "single_request_decode_tokens_per_second"
                            ],
                            "latency_p50_ms": profile["latency_p50_ms"],
                            "latency_p95_ms": profile["latency_p95_ms"],
                            "requests_per_second": profile["requests_per_second"],
                            "prompt_tokens_total": profile["prompt_tokens_total"],
                            "completion_tokens_total": profile["completion_tokens_total"],
                        }
                        for profile in model["profiles"]
                    ],
                    "cold_startup": model["cold_startup"],
                    "warm_startup": model["warm_startup"],
                }
                for model in suite["models"]
            ],
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    except Exception as exc:
        FAILURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FAILURE_PATH.write_text(
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}\n",
            encoding="utf-8",
        )
        print(f"benchmark_failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
