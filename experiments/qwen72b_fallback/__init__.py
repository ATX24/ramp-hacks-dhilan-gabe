"""Isolated Qwen2.5-72B-Instruct fallback / teacher workstream.

Two precise roles for the same pinned base snapshot:

1. ``teacher`` — powerful teacher supervising smaller TinyFable tiers.
2. ``oracle_sft_adapted_fallback`` — post-trained 72B finance fallback via
   synthetic oracle / sequence SFT on the 72B base itself.

The adapted 72B fallback is never labeled a distilled student unless a
separately identified larger teacher supplies its supervision.
"""

from __future__ import annotations

QWEN72B_PROFILE_NAME = "qwen72b_oracle_sft_fallback_v1"
TEACHER_ROLE_NAME = "teacher"
FALLBACK_ROLE_NAME = "oracle_sft_adapted_fallback"
# Affirmative phrases only. Schema keys like is_distilled_student=false are allowed.
FORBIDDEN_STUDENT_CLAIMS = frozenset(
    {
        "is a distilled student",
        "distilled student 72b",
        "72b is a distilled student",
        "72b distilled student",
        "student of larger teacher",
        "kd student 72b",
    }
)

__all__ = [
    "FALLBACK_ROLE_NAME",
    "FORBIDDEN_STUDENT_CLAIMS",
    "QWEN72B_PROFILE_NAME",
    "TEACHER_ROLE_NAME",
]
