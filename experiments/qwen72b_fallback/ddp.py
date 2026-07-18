"""DDP helpers: rank-isolated logs, failure propagation, rank-safe save/reload."""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class ProcessGroup(Protocol):
    def barrier(self) -> None: ...

    def abort(self) -> None: ...

    def shutdown(self) -> None: ...

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...


class RankFailure(RuntimeError):
    def __init__(self, rank: int, message: str) -> None:
        self.rank = rank
        super().__init__(f"rank {rank} failed: {message}")


@dataclass
class RankLogWriter:
    log_dir: Path
    rank: int
    _handle: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"rank-{self.rank:02d}.jsonl"
        self._handle = path.open("a", encoding="utf-8")

    def write(self, event: str, **payload: Any) -> None:
        row = {"rank": self.rank, "event": event, **payload}
        self._handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


@dataclass
class FailureBus:
    rank: int
    world_size: int
    failures: dict[int, str] = field(default_factory=dict)

    def report(self, message: str) -> None:
        self.failures[self.rank] = message

    def raise_if_any(self) -> None:
        if not self.failures:
            return
        ordered = sorted(self.failures.items())
        details = "; ".join(f"rank{rank}: {msg}" for rank, msg in ordered)
        raise RankFailure(ordered[0][0], details)


class FakeProcessGroup:
    def __init__(self, rank: int, world_size: int) -> None:
        self._rank = rank
        self._world_size = world_size
        self.barrier_calls = 0
        self.abort_calls = 0
        self.shutdown_calls = 0
        self._aborted = False

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def barrier(self) -> None:
        if self._aborted:
            raise RuntimeError("barrier after abort")
        self.barrier_calls += 1

    def abort(self) -> None:
        self.abort_calls += 1
        self._aborted = True

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def run_with_failure_propagation(
    *,
    group: ProcessGroup,
    bus: FailureBus,
    logger: RankLogWriter,
    body: Callable[[], None],
) -> None:
    try:
        body()
        group.barrier()
        bus.raise_if_any()
    except Exception as exc:  # noqa: BLE001 - propagate every rank failure loudly
        bus.report(str(exc))
        logger.write(
            "rank_failure",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        try:
            group.abort()
        finally:
            group.shutdown()
        raise RankFailure(group.rank, str(exc)) from exc
    else:
        group.shutdown()
        logger.write("rank_clean_shutdown")


@dataclass(frozen=True, slots=True)
class RankSafeArtifactPlan:
    """Only rank-0 writes sealed artifacts; all ranks barrier before/after."""

    adapter_dir: Path
    reload_probe_path: Path
    manifest_path: Path
    writer_rank: int = 0

    def may_write(self, rank: int) -> bool:
        return rank == self.writer_rank


def assert_rank_safe_reload(
    *,
    rank: int,
    plan: RankSafeArtifactPlan,
    adapter_files_present: bool,
    reload_ok: bool,
) -> None:
    if rank == plan.writer_rank:
        if not adapter_files_present:
            raise RuntimeError("rank-0 adapter save missing before reload probe")
        if not reload_ok:
            raise RuntimeError("rank-0 adapter reload probe failed")
    elif adapter_files_present and rank != plan.writer_rank:
        # Non-writer ranks must not race writes; presence is ok only after barrier.
        return
