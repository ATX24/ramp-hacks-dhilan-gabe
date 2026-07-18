"""DDP sampler hashes, divergence, duplicates, and failure propagation."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.huge_backup.ddp import (
    FailureBus,
    FakeProcessGroup,
    RankFailure,
    RankLogWriter,
    run_with_failure_propagation,
)
from experiments.huge_backup.sampler import (
    SamplerError,
    assert_plans_equal,
    assert_rank_order_matches,
    build_sampler_plan,
)


def test_rejects_duplicate_samples() -> None:
    with pytest.raises(SamplerError, match="duplicate"):
        build_sampler_plan(["a", "b", "a", "c"], world_size=2, seed=17)


def test_rank_divergence_detected() -> None:
    left = build_sampler_plan(["a", "b", "c", "d"], world_size=2, seed=17)
    right = build_sampler_plan(["a", "b", "c", "d"], world_size=2, seed=18)
    with pytest.raises(SamplerError, match="divergence"):
        assert_plans_equal(left, right)


def test_sampler_mismatch_on_rank() -> None:
    plan = build_sampler_plan([f"e{i}" for i in range(8)], world_size=4, seed=17)
    with pytest.raises(SamplerError, match="sampler mismatch"):
        assert_rank_order_matches(plan, rank=1, local_ids=["wrong"])


def test_partial_rank_failure_propagates(tmp_path: Path) -> None:
    group = FakeProcessGroup(rank=3, world_size=8)
    bus = FailureBus(rank=3, world_size=8)
    logger = RankLogWriter(tmp_path / "logs", rank=3)

    def body() -> None:
        raise RuntimeError("cuda oom on rank 3")

    with pytest.raises(RankFailure, match="rank 3 failed"):
        run_with_failure_propagation(group=group, bus=bus, logger=logger, body=body)
    assert group.abort_calls == 1
    assert group.shutdown_calls == 1
    logger.close()
    log_text = (tmp_path / "logs" / "rank-03.jsonl").read_text(encoding="utf-8")
    assert "rank_failure" in log_text


def test_clean_shutdown_writes_rank_log(tmp_path: Path) -> None:
    group = FakeProcessGroup(rank=0, world_size=2)
    bus = FailureBus(rank=0, world_size=2)
    logger = RankLogWriter(tmp_path / "logs", rank=0)
    run_with_failure_propagation(group=group, bus=bus, logger=logger, body=lambda: None)
    logger.close()
    assert "rank_clean_shutdown" in (tmp_path / "logs" / "rank-00.jsonl").read_text()
