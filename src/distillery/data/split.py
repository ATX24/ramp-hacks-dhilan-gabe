"""Grouped split isolation for IID and OOD finance-world corpora."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from distillery.contracts.tasks import SplitName


@dataclass(frozen=True)
class SplitSpec:
    name: SplitName
    count: int
    token: str
    ood: bool = False


SMOKE_SPLITS: tuple[SplitSpec, ...] = (
    SplitSpec(SplitName.TRAIN, 320, "smk_tr", ood=False),
    SplitSpec(SplitName.VALIDATION, 80, "smk_va", ood=False),
    SplitSpec(SplitName.TEST, 160, "smk_te", ood=False),
)

FULL_SPLITS: tuple[SplitSpec, ...] = (
    SplitSpec(SplitName.TRAIN, 3200, "full_tr", ood=False),
    SplitSpec(SplitName.VALIDATION, 400, "full_va", ood=False),
    SplitSpec(SplitName.IID_TEST, 800, "full_iid", ood=False),
    SplitSpec(SplitName.OOD_TEST, 800, "full_ood", ood=True),
)


def isolation_keys(example_input: dict, world_id: str, group_id: str) -> frozenset[str]:
    """Identity keys that must remain disjoint across non-OOD-compatible splits."""
    keys = {
        f"world:{world_id}",
        f"group:{group_id}",
    }
    for field in ("entity_id", "txn_id", "close_period"):
        value = example_input.get(field)
        if value:
            keys.add(f"{field}:{value}")
    for entry in example_input.get("book_entries") or []:
        if isinstance(entry, Mapping) and entry.get("id"):
            keys.add(f"book:{entry['id']}")
    for event in example_input.get("bank_events") or []:
        if isinstance(event, Mapping) and event.get("id"):
            keys.add(f"bank:{event['id']}")
    return frozenset(keys)
