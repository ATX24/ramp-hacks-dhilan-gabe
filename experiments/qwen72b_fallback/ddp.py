"""Bounded DDP synchronization with one visible GPU per child process."""

from __future__ import annotations

import json
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, TypeVar

T = TypeVar("T")


class RankFailure(RuntimeError):
    pass


@dataclass
class RankLogWriter:
    log_dir: Path
    rank: int
    _handle: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._handle = (self.log_dir / f"rank-{self.rank:02d}.jsonl").open(
            "a",
            encoding="utf-8",
        )

    def write(self, event: str, **payload: Any) -> None:
        self._handle.write(
            json.dumps(
                {"rank": self.rank, "event": event, **payload},
                sort_keys=True,
            )
            + "\n"
        )
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: Literal[0]
    device_index: Literal[0]
    timeout_seconds: int
    torch: Any
    dist: Any

    @property
    def is_writer(self) -> bool:
        return self.rank == 0


def initialize_distributed(
    *,
    timeout_seconds: int = 120,
    torch_module: Any | None = None,
) -> DistributedContext:
    if torch_module is None:
        import torch as torch_module

    required = {"RANK", "WORLD_SIZE", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES"}
    missing = sorted(required - set(os.environ))
    if missing:
        raise RuntimeError(f"distributed child environment is incomplete: {missing}")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    if world_size != 8:
        raise RuntimeError("Qwen72B DDP requires exactly eight ranks")
    if local_rank != 0:
        raise RuntimeError("isolated DDP child must use logical LOCAL_RANK=0")
    if not visible.isascii() or not visible.isdecimal() or "," in visible:
        raise RuntimeError("each DDP child must expose exactly one physical GPU")
    if int(torch_module.cuda.device_count()) != 1:
        raise RuntimeError("each DDP child must see exactly one logical CUDA GPU")
    torch_module.cuda.set_device(0)
    dist = torch_module.distributed
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )
    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=0,
        device_index=0,
        timeout_seconds=timeout_seconds,
        torch=torch_module,
        dist=dist,
    )


def run_synchronized_phase(
    context: DistributedContext,
    phase: str,
    operation: Callable[[], T],
) -> T:
    """All ranks report local success/failure; NCCL timeout bounds rank death."""
    sentinel = object()
    result: T | object = sentinel
    local_error: str | None = None
    local_traceback: str | None = None
    try:
        result = operation()
    except BaseException as exc:  # noqa: BLE001 - report all rank failures
        local_error = f"{type(exc).__name__}: {exc}"
        local_traceback = traceback.format_exc()
    payload = {
        "rank": context.rank,
        "phase": phase,
        "error": local_error,
        "traceback": local_traceback,
    }
    gathered: list[dict[str, Any] | None] = [None] * context.world_size
    try:
        context.dist.all_gather_object(gathered, payload)
    except BaseException as exc:  # noqa: BLE001 - normalize bounded collective failures
        raise RankFailure(
            f"bounded synchronization failed in {phase}; a rank may have died: {exc}"
        ) from exc
    failures = [item for item in gathered if isinstance(item, dict) and item.get("error")]
    if failures:
        summary = "; ".join(f"rank {item['rank']}: {item['error']}" for item in failures)
        raise RankFailure(f"distributed phase {phase} failed: {summary}")
    if result is sentinel:
        raise RankFailure(f"distributed phase {phase} produced no local result")
    return result


def require_equal_shapes(
    context: DistributedContext,
    shape: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    gathered: list[tuple[int, ...] | None] = [None] * context.world_size
    context.dist.all_gather_object(gathered, shape)
    if any(item != shape for item in gathered):
        raise RankFailure(f"cross-rank batch shape mismatch: {gathered}")
    return tuple(item for item in gathered if item is not None)


def acknowledge_all_ranks(context: DistributedContext) -> tuple[bool, ...]:
    acknowledgements: list[dict[str, Any] | None] = [None] * context.world_size
    context.dist.all_gather_object(
        acknowledgements,
        {"rank": context.rank, "complete": True},
    )
    ranks = {
        int(item["rank"])
        for item in acknowledgements
        if isinstance(item, dict) and item.get("complete") is True
    }
    if ranks != set(range(context.world_size)):
        raise RankFailure(f"all-rank completion acknowledgement failed: {ranks}")
    return tuple(True for _ in range(context.world_size))


def shutdown_distributed(context: DistributedContext) -> None:
    """Destroy without a final barrier, which could deadlock after rank death."""
    if context.dist.is_initialized():
        context.dist.destroy_process_group()
