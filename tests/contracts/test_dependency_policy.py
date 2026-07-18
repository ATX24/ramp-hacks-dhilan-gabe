"""Foundation/ML dependency isolation and image compatibility locks."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_ml_extra_is_exact_and_linux_reuses_image_torch() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["optional-dependencies"]["ml"] == [
        "torch==2.4.1; sys_platform != 'linux'",
        "transformers==4.46.3",
        "peft==0.13.2",
        "bitsandbytes==0.44.1",
        "accelerate==1.1.1",
        "safetensors==0.4.5",
    ]
    assert project["tool"]["uv"]["override-dependencies"] == [
        "torch==2.4.1; sys_platform != 'linux'"
    ]


def test_lock_has_no_linux_cuda_overlay_stack() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {package["name"]: package["version"] for package in lock["package"]}
    assert packages["torch"] == "2.4.1"
    assert packages["transformers"] == "4.46.3"
    assert packages["peft"] == "0.13.2"
    assert packages["bitsandbytes"] == "0.44.1"
    assert packages["accelerate"] == "1.1.1"
    assert packages["safetensors"] == "0.4.5"
    assert not any(name.startswith("nvidia-") for name in packages)
    assert "triton" not in packages
    assert lock["manifest"]["overrides"] == [
        {
            "name": "torch",
            "marker": "sys_platform != 'linux'",
            "specifier": "==2.4.1",
        }
    ]
