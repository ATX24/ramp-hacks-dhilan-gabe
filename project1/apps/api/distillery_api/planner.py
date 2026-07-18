"""plan_distillation(): pure with respect to training. Inspects metadata, checks
gates, resolves the recipe transparently, and estimates cost. Launches nothing."""
from __future__ import annotations

import os

from distillery.contracts.recipes import RecipeResolution
from distillery.recipes.auto import resolve
from distillery.synthesis.teacher import TEACHER_MODEL, TEACHER_PARAMS

STUDENT = {"id": "Qwen/Qwen2.5-0.5B-Instruct",
           "revision": "pinned-at-runtime",  # resolved to a commit SHA by preflight download
           "access": "white_box"}
TEACHER = {"id": TEACHER_MODEL, "revision": TEACHER_MODEL, "access": "api_black_box"}

# Opus pricing (per million tokens) for teacher-cost estimation.
OPUS_IN_PER_M = 15.0
OPUS_OUT_PER_M = 75.0


def plan(dataset_meta: dict, requested_recipe: str = "auto",
         max_run_usd: float = 25.0) -> dict:
    counts = dataset_meta.get("counts_by_label_source", {})
    n_train = dataset_meta.get("counts_by_split", {}).get("train", 0) + \
              dataset_meta.get("counts_by_split", {}).get("validation", 0)
    has_valid = counts.get("teacher", 0) + counts.get("imported", 0) >= n_train and n_train > 0
    teacher_available = bool(os.environ.get("ANTHROPIC_API_KEY"))

    resolution: RecipeResolution = resolve(
        requested_recipe,
        has_valid_responses=has_valid,
        teacher_access=TEACHER["access"],
        tokenizers_match=False,       # Opus (API) vs Qwen student: never logit-compatible
        memory_dry_run_ok=False,      # no white-box teacher to probe
        teacher_available=teacher_available,
    )

    # Teacher synthesis cost estimate: ~900 input + ~250 output tokens per example.
    calls = 0 if has_valid else n_train
    est_teacher = calls * (900 * OPUS_IN_PER_M + 250 * OPUS_OUT_PER_M) / 1_000_000
    est_low, est_high = round(est_teacher * 0.7, 2), round(est_teacher * 1.5 + 2.0, 2)

    blockers = []
    if not teacher_available and not has_valid and resolution.resolved == "sequence.v1":
        blockers.append({"code": "TEACHER_UNAVAILABLE",
                         "message": "ANTHROPIC_API_KEY not configured on the server."})
    if est_high > max_run_usd:
        blockers.append({"code": "ESTIMATED_BUDGET_EXCEEDED",
                         "message": f"High estimate ${est_high} exceeds ceiling ${max_run_usd}."})

    return {
        "models": {"teacher": TEACHER, "student": STUDENT},
        "tokenizer_fingerprints": {
            "logit_kd_compatible": False,
            "reason": "Teacher is Claude Opus via API (black-box); logit.v1 requires a local "
                      "white-box teacher with an exactly matching tokenizer.",
        },
        "capabilities": {"local": True, "sagemaker": False,
                         "notes": "This deployment runs synthesis + curation; training executes "
                                  "via the same manifest on a GPU host."},
        "dataset": {k: dataset_meta.get(k) for k in
                    ("dataset_id", "sha256", "split_sha256", "counts_by_task",
                     "counts_by_split", "counts_by_label_source")},
        "license": {"student": "Apache-2.0 (Qwen2.5-0.5B-Instruct)",
                    "teacher_output_use": "Anthropic API outputs used for internal model "
                                          "training per applicable terms; reviewed at preflight."},
        "recipe": resolution.model_dump(),
        "teacher_synthesis": {"model": TEACHER_MODEL, "params": TEACHER_PARAMS,
                              "estimated_calls": calls},
        "estimates": {"teacher_cost_low_usd": est_low, "teacher_cost_high_usd": est_high,
                      "assumptions": "900 input / 250 output tokens per call at Opus list price",
                      "gpu_wall_time": "45-90 min sequence SFT on one A10G (planning range)"},
        "blockers": blockers,
    }
