"""Tiny deterministic emergency subset: 32–64 train / 16 validation, no split leaks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillery.contracts.hashing import content_sha256, sha256_hex
from distillery.contracts.tasks import SplitName, TaskId
from distillery.data.generate import CorpusSpec, GeneratedCorpus, generate_corpus
from distillery.data.leakage import check_leakage
from distillery.data.split import SplitSpec
from experiments.aws_smoke.profile import DEFAULT_EMERGENCY_PROFILE, EmergencyTrainingProfile

# transaction_review prompts currently embed full policy/COA payloads and exceed
# the sealed 512-token emergency max_length. Keep the smoke corpus on short tasks.
_EMERGENCY_TASK_MIXTURE: dict[TaskId, float] = {
    TaskId.VARIANCE_ANALYSIS: 0.8,
    TaskId.CASH_RECONCILIATION: 0.2,
}
_EMERGENCY_TASK_ORDER: tuple[TaskId, ...] = (
    TaskId.VARIANCE_ANALYSIS,
    TaskId.CASH_RECONCILIATION,
)


def emergency_corpus_spec(profile: EmergencyTrainingProfile | None = None) -> CorpusSpec:
    p = profile or DEFAULT_EMERGENCY_PROFILE
    return CorpusSpec(
        name="smoke",
        seed=p.seed,
        splits=(
            SplitSpec(SplitName.TRAIN, p.train_examples, "emg_tr", ood=False),
            SplitSpec(SplitName.VALIDATION, p.validation_examples, "emg_va", ood=False),
        ),
        task_mixture=dict(_EMERGENCY_TASK_MIXTURE),
        task_order=_EMERGENCY_TASK_ORDER,
    )


@dataclass(frozen=True, slots=True)
class EmergencySubset:
    corpus: GeneratedCorpus
    train_path: Path
    validation_path: Path
    subset_manifest_path: Path
    content_sha256: str
    split_sha256: dict[str, str]


def materialize_emergency_subset(
    output_dir: Path,
    *,
    profile: EmergencyTrainingProfile | None = None,
) -> EmergencySubset:
    """Generate and write the emergency subset; fail loud on any leakage finding."""
    p = profile or DEFAULT_EMERGENCY_PROFILE
    corpus = generate_corpus(
        emergency_corpus_spec(p),
        validate=True,
        check_near_duplicates=True,
    )
    leakage = check_leakage(corpus.examples, check_near_duplicates=True)
    fatal = [
        f
        for f in leakage.findings
        if f.kind
        in {
            "exact_normalized_duplicate",
            "cross_split_id",
            "cross_split_template",
            "cross_split_near_duplicate",
        }
    ]
    if fatal:
        raise ValueError(
            f"emergency subset leakage detected ({len(fatal)} findings); first={fatal[0]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = corpus.write(output_dir)
    train_path = output_dir / f"{SplitName.TRAIN.value}.jsonl"
    validation_path = output_dir / f"{SplitName.VALIDATION.value}.jsonl"
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("corpus.write did not produce train/validation jsonl")

    # Enforce no target/split leak once fixed files exist: re-hash and compare.
    train_ids = _example_ids(train_path)
    val_ids = _example_ids(validation_path)
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(f"train/validation example_id overlap: {sorted(overlap)[:5]}")

    content = content_sha256(
        [e.model_dump(mode="json") for e in corpus.examples]
    )
    subset_manifest: dict[str, Any] = {
        "schema_version": "distillery.aws_smoke.subset.v1",
        "profile": p.name,
        "seed": p.seed,
        "train_examples": p.train_examples,
        "validation_examples": p.validation_examples,
        "content_sha256": content,
        "split_sha256": {
            "train": hashes[train_path.name],
            "validation": hashes[validation_path.name],
        },
        "file_sha256": hashes,
        "leakage": leakage.to_dict(),
        "generator_manifest_sha256": corpus.manifest.get("manifest_sha256"),
    }
    subset_path = output_dir / "emergency_subset_manifest.json"
    text = json.dumps(subset_manifest, sort_keys=True, indent=2) + "\n"
    subset_path.write_text(text, encoding="utf-8")
    return EmergencySubset(
        corpus=corpus,
        train_path=train_path,
        validation_path=validation_path,
        subset_manifest_path=subset_path,
        content_sha256=content,
        split_sha256={
            "train": hashes[train_path.name],
            "validation": hashes[validation_path.name],
        },
    )


def load_fixed_subset_hashes(subset_manifest_path: Path) -> dict[str, str]:
    raw = json.loads(subset_manifest_path.read_text(encoding="utf-8"))
    split_sha256 = raw.get("split_sha256")
    if not isinstance(split_sha256, dict):
        raise ValueError("subset manifest missing split_sha256")
    return {str(k): str(v) for k, v in split_sha256.items()}


def verify_fixed_subset_no_leak(
    *,
    train_path: Path,
    validation_path: Path,
    expected_split_sha256: dict[str, str],
) -> None:
    """Once fixed data exists, refuse silent drift or split leaks."""
    train_bytes = train_path.read_bytes()
    val_bytes = validation_path.read_bytes()
    actual = {
        "train": sha256_hex(train_bytes),
        "validation": sha256_hex(val_bytes),
    }
    for key, expected in expected_split_sha256.items():
        if actual.get(key) != expected:
            raise ValueError(
                f"fixed subset hash drift for {key}: "
                f"expected={expected} actual={actual.get(key)}"
            )
    overlap = set(_example_ids(train_path)) & set(_example_ids(validation_path))
    if overlap:
        raise ValueError(f"train/validation example_id overlap: {sorted(overlap)[:5]}")


def _example_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        example_id = payload.get("example_id")
        if not isinstance(example_id, str):
            raise ValueError(f"missing example_id in {path}")
        ids.append(example_id)
    return ids
