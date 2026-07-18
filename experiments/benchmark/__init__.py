"""Finite, tagged Transformers throughput benchmarks for sealed Qwen snapshots."""

from __future__ import annotations

SCHEMA_VERSION = "distillery.benchmark.systems.v1"
RUNTIME_LABEL = "transformers-4.46.3/torch-2.4.1/cuda-12.4"
ARTIFACT_LABEL = "base_model_proxy_until_trained_adapters"

__all__ = ["ARTIFACT_LABEL", "RUNTIME_LABEL", "SCHEMA_VERSION"]
