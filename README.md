# Distillery

**Smaller models. Proven economics.**

Distillery turns traces and synthetic finance examples into **TinyFable**, one portable finance-generalist student model, then proves whether distillation beats the base model, a cheap API, and rules on quality and economics.

This repository is the Ramp hackathon implementation. The foundation freeze (`contracts-v1`) lives in `src/distillery/contracts/` and `tests/fixtures/finance_world_v1/`.

## Requirements

- Python >= 3.11
- [`uv`](https://docs.astral.sh/uv/) (recommended)

## Install (foundation / contracts)

Heavy ML stacks (`torch`, `transformers`, etc.) are optional and **not** installed by default.

```bash
uv sync --extra dev
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To resolve the pinned ML compatibility set later (trainers only):

```bash
uv sync --extra dev --extra ml
```

The ML extra is locked to PyTorch 2.4.1, Transformers 4.46.3, PEFT 0.13.2,
bitsandbytes 0.44.1, Accelerate 1.1.1, and safetensors 0.4.5. This matches the
PyTorch 2.4 / CUDA 12.4 A10G training image. The image owns its CUDA-enabled
`torch==2.4.1+cu124` build; image setup must retain and verify that build rather
than layering a different PyPI Torch/CUDA stack over it. On Linux, the `ml`
extra therefore overrides transitive Torch requirements and installs no Torch
or NVIDIA CUDA wheels; it must run in that pre-provisioned image. Non-Linux
development resolves the API-compatible CPU package at `torch==2.4.1`.

## Contract / foundation tests

```bash
uv run pytest tests/contracts -q
```

Golden fixtures: `tests/fixtures/finance_world_v1/` (exactly 12 envelopes + oracle expectations + frozen SHA-256 manifest). The manifest records separate RFC 8785 semantic-example hashes, raw JSONL line hashes (including newline bytes), and whole-file hashes.

## Public resources (frozen)

- `Dataset`
- `DistillationRun`
- `ModelArtifact`
- `ProofReport`

Implemented recipes: `sequence.v1`, `logit.v1`. `auto` is a transparent resolver. Its locked precedence starts with `do_not_distill` when a cheaper baseline is already known to satisfy both quality and economics gates. This is intentional: the locked baseline gate runs before any billable training decision, even though earlier numbered prose was ambiguous. It then considers existing responses, compatible white-box logit transfer, and an allowed teacher in that order. Explicit recipes never downgrade. Catalog-only and unknown methods fail with `RECIPE_NOT_IMPLEMENTED`.

`logit.v1` manifests seal exact tokenizer, special-token, chat-template, memory-probe, runtime-image, resolver, and student-tokenizer completion-count evidence. Unknown capability values never count as passing evidence.

## Scope notes

Foundation work does not download model weights, start training, submit SageMaker jobs, or deploy.
