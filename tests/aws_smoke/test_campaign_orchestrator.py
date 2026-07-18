"""Process isolation, cleanup, timeout, determinism, and cost tests."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from experiments.aws_smoke.campaign_orchestrator import (
    allocate_parent_cost,
    build_child_environment,
    parent_cost_microusd,
    run_campaign,
)
from experiments.aws_smoke.pins import EmergencyEvidence
from tests.aws_smoke.campaign_support import (
    G5_12_PRICE_MICROUSD,
    G5_48_PRICE_MICROUSD,
    stage_test_campaign,
)


class FakeProcess:
    def __init__(
        self,
        pid: int,
        polls: list[int | None],
        *,
        timeout_once: bool = False,
    ) -> None:
        self.pid = pid
        self._polls = list(polls)
        self._last: int | None = None
        self._timeout_once = timeout_once

    def poll(self) -> int | None:
        if self._polls:
            self._last = self._polls.pop(0)
        return self._last

    def wait(self, timeout: float | None = None) -> int:
        if self._timeout_once:
            self._timeout_once = False
            raise subprocess.TimeoutExpired("fake", timeout)
        self._last = -signal.SIGTERM
        return self._last


class FakePopenFactory:
    def __init__(
        self,
        scripts: list[list[int | None]],
        *,
        fail_at: int | None = None,
        timeout_once: bool = False,
    ) -> None:
        self.scripts = scripts
        self.fail_at = fail_at
        self.timeout_once = timeout_once
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.processes: list[FakeProcess] = []

    def __call__(self, command: list[str], **kwargs: Any) -> FakeProcess:
        index = len(self.calls)
        self.calls.append((command, kwargs))
        if self.fail_at == index:
            raise OSError("synthetic startup failure")
        process = FakeProcess(
            10_000 + index,
            self.scripts[index],
            timeout_once=self.timeout_once,
        )
        self.processes.append(process)
        return process


def _paths(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset = tmp_path / "dataset"
    models = tmp_path / "models"
    dataset.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    return {
        "dataset_dir": dataset.resolve(),
        "models_dir": models.resolve(),
        "output_root": (tmp_path / "output").resolve(),
        "model_root": (tmp_path / "model").resolve(),
        "runtime_root": (tmp_path / "runtime").resolve(),
    }


def _run(
    *,
    bundle: Any,
    tmp_path: Path,
    factory: FakePopenFactory,
    monotonic: Callable[[], float] = lambda: 0.0,
    monotonic_ns: Callable[[], int] | None = None,
    signaler: Callable[[int, int], None] = lambda _pid, _signal: None,
    gpu_count: int | None = None,
) -> Any:
    nanosecond_values = iter([0, 3_600_000_000_000])
    return run_campaign(
        campaign_root=bundle.root,
        expected_index_sha256=bundle.index_sha256,
        interpreter=Path(sys.executable).resolve(),
        parent_env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": "/opt/distillery/src",
            "AWS_SECRET_ACCESS_KEY": "must-not-leak",
        },
        popen_factory=factory,
        artifact_verifier=lambda _path: {"ok": True},
        process_group_signaler=signaler,
        gpu_count_provider=lambda: (
            bundle.index.hardware.gpu_count if gpu_count is None else gpu_count
        ),
        monotonic=monotonic,
        monotonic_ns=(
            monotonic_ns if monotonic_ns is not None else lambda: next(nanosecond_values)
        ),
        sleep=lambda _seconds: None,
        **_paths(tmp_path),
    )


def test_eight_children_are_isolated_and_costed_once(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    bundle = stage_test_campaign(
        tmp_path / "bundle-data",
        valid_evidence,
        profile_id="g5-48xlarge-8xa10g-independent-v1",
        hourly_price_microusd=G5_48_PRICE_MICROUSD,
    )
    factory = FakePopenFactory([[None, 0] for _ in range(8)])
    result = _run(
        bundle=bundle,
        tmp_path=tmp_path / "runtime-data",
        factory=factory,
    )
    assert result.status == "succeeded"
    assert result.parent_cost_microusd == G5_48_PRICE_MICROUSD
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["allocation_total_microusd"] == G5_48_PRICE_MICROUSD
    assert (
        sum(child["allocated_cost_microusd"] for child in summary["children"])
        == G5_48_PRICE_MICROUSD
    )
    inventory = json.loads(result.inventory_path.read_text(encoding="utf-8"))
    preserved_manifests = {
        record["path"]
        for record in inventory["files"]
        if record["path"].endswith("/sealed_manifest.json")
    }
    assert len(preserved_manifests) == 8
    assert all(path.startswith("output/arms/") for path in preserved_manifests)
    assert len(factory.calls) == 8

    output_paths: set[str] = set()
    model_paths: set[str] = set()
    working_directories: set[str] = set()
    for slot, (command, kwargs) in enumerate(factory.calls):
        assert Path(command[0]).is_absolute()
        assert kwargs["start_new_session"] is True
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == str(slot)
        assert kwargs["env"]["DISTILLERY_CAMPAIGN_SINGLE_GPU"] == "1"
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert "AWS_SECRET_ACCESS_KEY" not in kwargs["env"]
        output_paths.add(command[command.index("--output-dir") + 1])
        model_paths.add(command[command.index("--model-output-dir") + 1])
        working_directories.add(str(kwargs["cwd"]))
    assert len(output_paths) == len(model_paths) == len(working_directories) == 8


def test_startup_failure_terminates_started_siblings(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    bundle = stage_test_campaign(tmp_path / "bundle-data", valid_evidence)
    factory = FakePopenFactory(
        [[None, None, None] for _ in range(4)],
        fail_at=1,
    )
    signals: list[tuple[int, int]] = []
    result = _run(
        bundle=bundle,
        tmp_path=tmp_path / "runtime-data",
        factory=factory,
        signaler=lambda pid, signum: signals.append((pid, signum)),
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.status == "failed"
    assert summary["children"][1]["status"] == "startup_failed"
    assert summary["children"][0]["status"] == "terminated"
    assert signals == [(10_000, signal.SIGTERM)]
    assert all(child["status"] == "not_started" for child in summary["children"][2:])


def test_runtime_failure_terminates_other_process_groups(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    bundle = stage_test_campaign(tmp_path / "bundle-data", valid_evidence)
    factory = FakePopenFactory([[None, 7]] + [[None, None, None, None] for _ in range(3)])
    signals: list[tuple[int, int]] = []
    result = _run(
        bundle=bundle,
        tmp_path=tmp_path / "runtime-data",
        factory=factory,
        signaler=lambda pid, signum: signals.append((pid, signum)),
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert result.status == "failed"
    assert summary["children"][0]["status"] == "failed"
    assert {pid for pid, signum in signals if signum == signal.SIGTERM} == {
        10_001,
        10_002,
        10_003,
    }
    assert all(child["status"] == "terminated" for child in summary["children"][1:])


def test_timeout_uses_bounded_term_then_kill(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    bundle = stage_test_campaign(tmp_path / "bundle-data", valid_evidence)
    factory = FakePopenFactory(
        [[None, None, None, None] for _ in range(4)],
        timeout_once=True,
    )
    times = iter([0.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0])
    signals: list[tuple[int, int]] = []
    result = _run(
        bundle=bundle,
        tmp_path=tmp_path / "runtime-data",
        factory=factory,
        monotonic=lambda: next(times, 21.0),
        signaler=lambda pid, signum: signals.append((pid, signum)),
    )
    assert result.status == "timed_out"
    assert {signum for _, signum in signals} == {
        signal.SIGTERM,
        signal.SIGKILL,
    }


def test_summaries_and_allocations_are_deterministic(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    summaries: list[bytes] = []
    for index in range(2):
        root = tmp_path / f"attempt-{index}"
        bundle = stage_test_campaign(root / "bundle-data", valid_evidence)
        factory = FakePopenFactory([[None, 0] for _ in range(4)])
        result = _run(
            bundle=bundle,
            tmp_path=root / "runtime-data",
            factory=factory,
        )
        summaries.append(result.summary_path.read_bytes())
    assert summaries[0] == summaries[1]


def test_existing_campaign_output_fails_closed_without_overwrite(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    bundle = stage_test_campaign(tmp_path / "bundle-data", valid_evidence)
    runtime_root = tmp_path / "runtime-data"
    first = FakePopenFactory([[None, 0] for _ in range(4)])
    assert _run(bundle=bundle, tmp_path=runtime_root, factory=first).status == "succeeded"
    second = FakePopenFactory([[None, 0] for _ in range(4)])
    with pytest.raises(FileExistsError, match="campaign path collision"):
        _run(bundle=bundle, tmp_path=runtime_root, factory=second)
    assert second.calls == []


def test_gpu_count_path_and_environment_mismatches_fail_closed(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    bundle = stage_test_campaign(tmp_path / "bundle-data", valid_evidence)
    factory = FakePopenFactory([[None, 0] for _ in range(4)])
    with pytest.raises(RuntimeError, match="GPU count"):
        _run(
            bundle=bundle,
            tmp_path=tmp_path / "runtime-data",
            factory=factory,
            gpu_count=8,
        )
    with pytest.raises(ValueError, match="non-absolute"):
        build_child_environment(
            {"PATH": "relative/bin"},
            gpu_slot=0,
            seed=17,
            failure_path=tmp_path / "failure",
        )


def test_integer_cost_math_never_multiplies_by_gpu_count(
    valid_evidence: EmergencyEvidence,
    tmp_path: Path,
) -> None:
    bundle = stage_test_campaign(tmp_path, valid_evidence)
    cost = parent_cost_microusd(
        hourly_price_microusd=G5_12_PRICE_MICROUSD,
        elapsed_ns=1_800_000_000_000,
    )
    allocations = allocate_parent_cost(cost, bundle.index.arms)
    assert cost == 3_545_000
    assert sum(allocations) == cost
    assert max(allocations) - min(allocations) <= 1
