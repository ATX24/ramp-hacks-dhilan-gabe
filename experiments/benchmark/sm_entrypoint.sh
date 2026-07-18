#!/usr/bin/env bash
# SageMaker Training Job entrypoint for Transformers throughput benchmarks.
set -euo pipefail

export PYTHONPATH="/opt/ml/input/data/code:/opt/distillery/src:/opt/distillery"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

INSTANCE_TYPE="${DISTILLERY_BENCHMARK_INSTANCE_TYPE:-ml.g5.xlarge}"
HARDWARE="${DISTILLERY_BENCHMARK_HARDWARE:-NVIDIA-A10G-24GB}"
DTYPE="${DISTILLERY_BENCHMARK_DTYPE:-bf16}"
MAX_NEW_TOKENS="${DISTILLERY_BENCHMARK_MAX_NEW_TOKENS:-128}"
WARMUPS="${DISTILLERY_BENCHMARK_WARMUPS:-20}"
TIMED="${DISTILLERY_BENCHMARK_TIMED:-200}"
BATCH_SIZES="${DISTILLERY_BENCHMARK_BATCH_SIZES:-1,8}"

exec python -m experiments.benchmark.run \
  --models-root /opt/ml/input/data/models \
  --output-dir /opt/ml/output/data \
  --dtype "${DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --warmups "${WARMUPS}" \
  --timed "${TIMED}" \
  --batch-sizes "${BATCH_SIZES}" \
  --hardware "${HARDWARE}" \
  --instance-type "${INSTANCE_TYPE}"
