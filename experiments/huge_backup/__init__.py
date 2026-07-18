"""Time-boxed huge-backup offline sequence distillation path.

Isolated from the emergency aws_smoke trainer and from exact logit KD.
Teacher responses (Qwen2.5-32B-Instruct) are materialized before the warm
timer; the student (Qwen2.5-14B-Instruct) trains with BF16 LoRA under
8-process single-node DDP on ml.p4de.24xlarge.
"""

from __future__ import annotations

HUGE_BACKUP_PROFILE_NAME = "huge_backup_offline_sequence_v1"
OBJECTIVE_MODE = "offline_sequence_distillation"
FORBIDDEN_OBJECTIVE_CLAIMS = frozenset(
    {
        "exact logit kd",
        "exact_logit_kd",
        "logit kd",
        "logit_kd",
        "full vocab kl",
        "full_vocab_kl",
        "online teacher logits",
        "online_teacher_logits",
    }
)

__all__ = [
    "FORBIDDEN_OBJECTIVE_CLAIMS",
    "HUGE_BACKUP_PROFILE_NAME",
    "OBJECTIVE_MODE",
]
