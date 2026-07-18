"""Campaign children must expose one logical GPU and use an explicit map."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from experiments.aws_smoke.device_mapping import model_device_map


def _torch(*, available: bool, count: int, current: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: count,
            current_device=lambda: current,
        )
    )


def test_campaign_uses_explicit_single_device_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISTILLERY_CAMPAIGN_SINGLE_GPU", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert model_device_map(_torch(available=True, count=1)) == {"": 0}


@pytest.mark.parametrize(
    ("visible", "available", "count", "current"),
    [
        ("0,1", True, 2, 0),
        ("all", True, 1, 0),
        ("0", False, 0, 0),
        ("0", True, 2, 0),
        ("0", True, 1, 1),
    ],
)
def test_campaign_gpu_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    visible: str,
    available: bool,
    count: int,
    current: int,
) -> None:
    monkeypatch.setenv("DISTILLERY_CAMPAIGN_SINGLE_GPU", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(RuntimeError):
        model_device_map(_torch(available=available, count=count, current=current))


def test_serial_flow_retains_existing_auto_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DISTILLERY_CAMPAIGN_SINGLE_GPU", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert model_device_map(_torch(available=False, count=0)) == "auto"
