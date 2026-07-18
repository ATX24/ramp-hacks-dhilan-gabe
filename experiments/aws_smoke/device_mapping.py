"""Explicit single-device mapping for isolated campaign trainer children."""

from __future__ import annotations

import os
from typing import Any

CAMPAIGN_SINGLE_GPU_ENV = "DISTILLERY_CAMPAIGN_SINGLE_GPU"


def model_device_map(torch_module: Any) -> str | dict[str, int]:
    """Keep serial behavior, but fail closed when campaign isolation is requested."""
    if os.environ.get(CAMPAIGN_SINGLE_GPU_ENV) != "1":
        return "auto"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.isascii() or not visible.isdecimal():
        raise RuntimeError("campaign trainer requires CUDA_VISIBLE_DEVICES to name exactly one GPU")
    if not torch_module.cuda.is_available():
        raise RuntimeError("campaign trainer requires CUDA")
    if int(torch_module.cuda.device_count()) != 1:
        raise RuntimeError("campaign trainer must see exactly one logical GPU")
    if int(torch_module.cuda.current_device()) != 0:
        raise RuntimeError("campaign trainer logical CUDA device must be zero")
    return {"": 0}
