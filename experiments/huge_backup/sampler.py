"""DDP-safe deterministic sampler with sealed order hashes."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from distillery.contracts.hashing import content_sha256


class SamplerError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SamplerPlan:
    example_ids: tuple[str, ...]
    world_size: int
    seed: int
    order_sha256: str
    per_rank_sha256: tuple[str, ...]

    def rank_ids(self, rank: int) -> tuple[str, ...]:
        if rank < 0 or rank >= self.world_size:
            raise SamplerError(f"rank {rank} out of range for world_size={self.world_size}")
        return tuple(
            self.example_ids[index] for index in range(rank, len(self.example_ids), self.world_size)
        )


def _stable_shuffle(ids: Sequence[str], *, seed: int) -> list[str]:
    # Deterministic Fisher-Yates using SHA-256 mix (no numpy dependency).
    items = list(ids)
    for index in range(len(items) - 1, 0, -1):
        digest = hashlib.sha256(f"{seed}:{index}:{items[index]}".encode()).digest()
        swap = int.from_bytes(digest[:8], "big") % (index + 1)
        items[index], items[swap] = items[swap], items[index]
    return items


def build_sampler_plan(
    example_ids: Sequence[str],
    *,
    world_size: int,
    seed: int,
    expected_count: int | None = None,
) -> SamplerPlan:
    if world_size < 1:
        raise SamplerError("world_size must be >= 1")
    if not example_ids:
        raise SamplerError("example_ids must be nonempty")
    if len(set(example_ids)) != len(example_ids):
        dupes = sorted({eid for eid in example_ids if example_ids.count(eid) > 1})
        raise SamplerError(f"duplicate example_ids forbidden: {dupes[:5]}")
    if expected_count is not None and len(example_ids) != expected_count:
        raise SamplerError(
            f"example count mismatch: expected={expected_count} actual={len(example_ids)}"
        )
    if len(example_ids) % world_size != 0:
        raise SamplerError(
            "example count must be divisible by world_size for DDP-equal shards "
            f"({len(example_ids)} % {world_size} != 0)"
        )
    ordered = tuple(_stable_shuffle(sorted(example_ids), seed=seed))
    per_rank = []
    for rank in range(world_size):
        rank_ids = tuple(ordered[index] for index in range(rank, len(ordered), world_size))
        per_rank.append(content_sha256({"rank": rank, "example_ids": list(rank_ids)}))
    order_sha = content_sha256(
        {
            "seed": seed,
            "world_size": world_size,
            "example_ids": list(ordered),
            "per_rank_sha256": per_rank,
        }
    )
    return SamplerPlan(
        example_ids=ordered,
        world_size=world_size,
        seed=seed,
        order_sha256=order_sha,
        per_rank_sha256=tuple(per_rank),
    )


def assert_rank_order_matches(plan: SamplerPlan, *, rank: int, local_ids: Sequence[str]) -> None:
    expected = plan.rank_ids(rank)
    if tuple(local_ids) != expected:
        raise SamplerError(
            f"rank {rank} sampler mismatch: expected_sha="
            f"{plan.per_rank_sha256[rank]} actual_sha="
            f"{content_sha256({'rank': rank, 'example_ids': list(local_ids)})}"
        )


def assert_plans_equal(left: SamplerPlan, right: SamplerPlan) -> None:
    if left.order_sha256 != right.order_sha256:
        raise SamplerError(
            "sampler order hash divergence across ranks: "
            f"{left.order_sha256} != {right.order_sha256}"
        )
    if left.per_rank_sha256 != right.per_rank_sha256:
        raise SamplerError("per-rank sampler hashes diverged")
