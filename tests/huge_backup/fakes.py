"""Tiny fake models and channel builders for huge_backup unit tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.huge_backup.channels import (
    CANONICAL_MANIFEST_FILENAME,
    CANONICAL_TEACHER_RESPONSES_FILENAME,
    OfflineChannels,
)
from experiments.huge_backup.profile import HugeBackupTrainingProfile
from experiments.huge_backup.provenance import build_teacher_record, write_teacher_responses
from experiments.huge_backup.sampler import build_sampler_plan


class TinyLinearModel:
    """Minimal trainable stand-in (no torch required)."""

    def __init__(self, dim: int = 4) -> None:
        self._weights = {f"w{index}": float(index + 1) for index in range(dim)}
        self._training = True

    def parameters(self) -> list[float]:
        return list(self._weights.values())

    def state_dict(self) -> dict[str, float]:
        return dict(self._weights)

    def load_state_dict(self, state: dict[str, float]) -> None:
        self._weights = dict(state)

    def train(self) -> None:
        self._training = True

    def eval(self) -> None:
        self._training = False


class TinyOptimizer:
    def __init__(self, model: TinyLinearModel, lr: float = 0.1) -> None:
        self.model = model
        self.lr = lr

    def step(self) -> None:
        for key, value in self.model._weights.items():
            self.model._weights[key] = value - self.lr

    def zero_grad(self) -> None:
        return None


def make_teacher_records(
    count: int,
    *,
    teacher_revision: str = "f" * 40,
) -> list[Any]:
    records = []
    for index in range(count):
        records.append(
            build_teacher_record(
                example_id=f"ex-{index:04d}",
                prompt_text=f"prompt-{index}",
                response_text=f"teacher-response-{index}",
                teacher_model_id="Qwen/Qwen2.5-32B-Instruct",
                teacher_revision=teacher_revision,
            )
        )
    return records


def materialize_channels(
    root: Path,
    profile: HugeBackupTrainingProfile,
    *,
    corrupt_teacher_hash: bool = False,
    include_label_key: bool = False,
    wrong_role: bool = False,
    duplicate_example: bool = False,
    logit_kd_claim: bool = False,
) -> tuple[OfflineChannels, str, str]:
    channels = OfflineChannels(
        root=root,
        manifest=root / "manifest",
        dataset=root / "dataset",
        models=root / "models",
        teacher_responses=root / "teacher_responses",
    )
    for path in (
        channels.manifest,
        channels.dataset,
        channels.models,
        channels.teacher_responses,
    ):
        path.mkdir(parents=True, exist_ok=True)

    records = make_teacher_records(profile.train_examples)
    if duplicate_example:
        records.append(records[0])
    teacher_path = channels.teacher_responses / CANONICAL_TEACHER_RESPONSES_FILENAME
    if include_label_key or wrong_role:
        payload = [row.model_dump(mode="json") for row in records]
        if include_label_key:
            payload[0]["expected_output"] = {"bad": True}
        if wrong_role:
            payload[0]["model_role"] = "student"
        teacher_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        teacher_sha = "0" * 64
    else:
        write_teacher_responses(teacher_path, records)
        from experiments.huge_backup.provenance import canonical_responses_sha256

        teacher_sha = canonical_responses_sha256(records)

    plan = build_sampler_plan(
        [row.example_id for row in records[: profile.train_examples]],
        world_size=profile.world_size,
        seed=profile.seed,
    )
    manifest = {
        "schema_version": "distillery.huge_backup.manifest.v1",
        "objective_mode": "offline_sequence_distillation",
        "train_examples": profile.train_examples,
        "teacher_responses_sha256": ("0" * 64) if corrupt_teacher_hash else teacher_sha,
        "sampler_order_sha256": plan.order_sha256,
        "torch_version": "2.4.1",
        "cuda_available": True,
        "flash_attn_importable": True,
        "student_model_id": profile.student_model_id,
        "teacher_model_id": profile.teacher_model_id,
    }
    if logit_kd_claim:
        manifest["notes"] = "this arm uses exact logit KD"
    (channels.manifest / CANONICAL_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (channels.dataset / "train.jsonl").write_text(
        "".join(
            json.dumps({"example_id": row.example_id, "input": {"x": 1}}) + "\n"
            for row in records[: profile.train_examples]
        ),
        encoding="utf-8",
    )
    return channels, teacher_sha, plan.order_sha256


def tiny_step(model: TinyLinearModel, optimizer: TinyOptimizer) -> float:
    optimizer.zero_grad()
    loss = sum(model.parameters()) / len(model.parameters())
    optimizer.step()
    return float(loss)
