"""Adversarial warm-path tests with mocked distributed runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.huge_backup.ddp import RankFailure
from experiments.huge_backup.profile import HugeBackupTrainingProfile
from experiments.huge_backup.protocol import ProtocolClaimError
from experiments.huge_backup.provenance import TeacherProvenanceError
from experiments.huge_backup.sampler import SamplerError
from experiments.huge_backup.train import WarmTrainConfig, prepare_sealed_state, run_rank
from tests.huge_backup.fakes import materialize_channels


def _config(
    tmp_path: Path,
    channels,
    profile: HugeBackupTrainingProfile,
    *,
    rank: int = 0,
) -> WarmTrainConfig:
    return WarmTrainConfig(
        channels=channels,
        output_root=tmp_path / f"out-{rank}",
        log_dir=tmp_path / "logs",
        rank=rank,
        world_size=profile.world_size,
        mode="warm",
        profile=profile,
    )


def test_warm_rank0_completes_and_writes_peft(
    tmp_path: Path,
    mini_profile: HugeBackupTrainingProfile,
) -> None:
    channels, _, _ = materialize_channels(tmp_path / "channels", mini_profile)
    result = run_rank(_config(tmp_path, channels, mini_profile, rank=0))
    assert result["completed_updates"] == mini_profile.max_updates
    assert result["artifacts"]["ok"] is True
    assert (tmp_path / "out-0" / "model" / "adapter" / "adapter_config.json").is_file()
    assert (tmp_path / "out-0" / "SHA256SUMS").is_file()


def test_bad_teacher_provenance_hash(
    tmp_path: Path,
    mini_profile: HugeBackupTrainingProfile,
) -> None:
    channels, _, _ = materialize_channels(
        tmp_path / "channels",
        mini_profile,
        corrupt_teacher_hash=True,
    )
    with pytest.raises(RankFailure) as excinfo:
        run_rank(_config(tmp_path, channels, mini_profile))
    assert "teacher_responses_sha256 mismatch" in str(excinfo.value)


def test_logit_kd_claim_rejected(
    tmp_path: Path,
    mini_profile: HugeBackupTrainingProfile,
) -> None:
    channels, _, _ = materialize_channels(
        tmp_path / "channels",
        mini_profile,
        logit_kd_claim=True,
    )
    with pytest.raises((RankFailure, ProtocolClaimError)):
        prepare_sealed_state(_config(tmp_path, channels, mini_profile))


def test_duplicate_samples_fail_closed(
    tmp_path: Path,
    mini_profile: HugeBackupTrainingProfile,
) -> None:
    channels, _, _ = materialize_channels(
        tmp_path / "channels",
        mini_profile,
        duplicate_example=True,
    )
    # duplicate breaks load_teacher_responses before sampler
    with pytest.raises((RankFailure, TeacherProvenanceError, SamplerError)):
        run_rank(_config(tmp_path, channels, mini_profile))


def test_partial_rank_failure(
    tmp_path: Path,
    mini_profile: HugeBackupTrainingProfile,
) -> None:
    channels, _, _ = materialize_channels(tmp_path / "channels", mini_profile)
    with pytest.raises(RankFailure, match="injected partial rank failure"):
        run_rank(
            _config(tmp_path, channels, mini_profile, rank=1),
            partial_fail_rank=1,
        )


def test_timeout_reserve_stops_training(
    tmp_path: Path,
    mini_profile: HugeBackupTrainingProfile,
) -> None:
    channels, _, _ = materialize_channels(tmp_path / "channels", mini_profile)

    class Clock:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            # First call seals started=0; later calls sit on the training deadline.
            return 0.0 if self.calls == 1 else 1470.0

    with pytest.raises(RankFailure, match="artifact reserve"):
        run_rank(_config(tmp_path, channels, mini_profile), clock=Clock())


def test_sampler_mismatch_across_peers(
    tmp_path: Path,
    mini_profile: HugeBackupTrainingProfile,
) -> None:
    channels, _, _ = materialize_channels(tmp_path / "channels", mini_profile)
    sealed = prepare_sealed_state(_config(tmp_path, channels, mini_profile))
    other = build_foreign_plan(mini_profile)
    with pytest.raises(RankFailure, match="divergence"):
        run_rank(
            _config(tmp_path, channels, mini_profile),
            peer_plans=[other],
        )
    assert sealed["sampler_plan"].order_sha256 != other.order_sha256


def build_foreign_plan(profile: HugeBackupTrainingProfile):
    from experiments.huge_backup.sampler import build_sampler_plan

    ids = [f"ex-{index:04d}" for index in range(profile.train_examples)]
    return build_sampler_plan(ids, world_size=profile.world_size, seed=profile.seed + 99)
