#!/usr/bin/env python3
"""Fail-loud lock and runtime checks for the pinned training image stack."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import tomllib
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

BLOCKED_EXIT = 3


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def package_by_name(lock: dict[str, Any], name: str) -> dict[str, Any] | None:
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ValueError("uv.lock has no package array")
    matches = [
        package for package in packages if isinstance(package, dict) and package.get("name") == name
    ]
    if len(matches) > 1:
        versions = sorted(str(package.get("version")) for package in matches)
        raise ValueError(f"uv.lock has multiple {name} versions: {versions}")
    return matches[0] if matches else None


def check_lock(lock_path: Path, compatibility_path: Path) -> dict[str, Any]:
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    compatibility = load_json_object(compatibility_path)
    torch_package = package_by_name(lock, "torch")
    bitsandbytes_package = package_by_name(lock, "bitsandbytes")
    expected_torch = compatibility["torch_version"]
    expected_bitsandbytes = compatibility["bitsandbytes_version"]
    reasons: list[str] = []

    actual_torch = str(torch_package.get("version")) if torch_package else None
    actual_bitsandbytes = str(bitsandbytes_package.get("version")) if bitsandbytes_package else None
    if actual_torch != expected_torch:
        reasons.append(f"torch lock mismatch: expected {expected_torch}, found {actual_torch}")
    if actual_bitsandbytes != expected_bitsandbytes:
        reasons.append(
            "bitsandbytes lock mismatch: "
            f"expected {expected_bitsandbytes}, found {actual_bitsandbytes}"
        )

    packages = lock.get("package", [])
    package_names = sorted(
        str(package["name"])
        for package in packages
        if isinstance(package, dict) and isinstance(package.get("name"), str)
    )
    forbidden_patterns = compatibility.get("forbidden_lock_package_patterns", [])
    forbidden_found = sorted(
        name
        for name in package_names
        if any(re.fullmatch(pattern, name) for pattern in forbidden_patterns)
    )
    if forbidden_found:
        reasons.append(
            "lock would layer an incompatible CUDA stack over the base: "
            + ", ".join(forbidden_found)
        )

    base_provided_packages = set(compatibility["base_provided_packages"])
    accelerator_packages = {
        name
        for name in package_names
        if name in {"torch", "triton"} or name.startswith("nvidia-") or name.startswith("cuda-")
    }
    unexpected_accelerator_packages = sorted(accelerator_packages - base_provided_packages)
    if unexpected_accelerator_packages:
        reasons.append(
            "lock contains accelerator packages not supplied by the immutable base: "
            + ", ".join(unexpected_accelerator_packages)
        )

    return {
        "schema_version": "distillery.training.ml-lock-check.v1",
        "status": "compatible" if not reasons else "blocked",
        "base_image": compatibility["base_image"],
        "expected": {
            "torch_version": expected_torch,
            "torch_cuda_version": compatibility["torch_cuda_version"],
            "cudnn_major": compatibility["cudnn_major"],
            "bitsandbytes_version": expected_bitsandbytes,
            "base_provided_packages": sorted(base_provided_packages),
        },
        "actual": {
            "torch_version": actual_torch,
            "bitsandbytes_version": actual_bitsandbytes,
            "forbidden_packages": forbidden_found,
            "base_provided_packages": sorted(accelerator_packages & base_provided_packages),
            "unexpected_accelerator_packages": unexpected_accelerator_packages,
        },
        "reasons": reasons,
    }


def check_runtime(compatibility_path: Path, *, require_bitsandbytes: bool) -> dict[str, Any]:
    compatibility = load_json_object(compatibility_path)
    torch = importlib.import_module("torch")
    actual_torch = str(torch.__version__).split("+", maxsplit=1)[0]
    actual_cuda = str(torch.version.cuda)
    cudnn_version = torch.backends.cudnn.version()
    actual_cudnn_major = int(str(cudnn_version)[0]) if cudnn_version else None
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    reasons: list[str] = []

    expected = {
        "python_version": compatibility["python_version"],
        "torch_version": compatibility["torch_version"],
        "torch_cuda_version": compatibility["torch_cuda_version"],
        "cudnn_major": compatibility["cudnn_major"],
        "bitsandbytes_version": compatibility["bitsandbytes_version"],
    }
    if actual_python != expected["python_version"]:
        reasons.append(
            f"python runtime mismatch: expected {expected['python_version']}, found {actual_python}"
        )
    if actual_torch != expected["torch_version"]:
        reasons.append(
            f"torch runtime mismatch: expected {expected['torch_version']}, found {actual_torch}"
        )
    if actual_cuda != expected["torch_cuda_version"]:
        reasons.append(
            f"CUDA runtime mismatch: expected {expected['torch_cuda_version']}, found {actual_cuda}"
        )
    if actual_cudnn_major != expected["cudnn_major"]:
        reasons.append(
            f"cuDNN runtime mismatch: expected major {expected['cudnn_major']}, "
            f"found {actual_cudnn_major}"
        )

    actual_bitsandbytes: str | None = None
    if require_bitsandbytes:
        bitsandbytes = importlib.import_module("bitsandbytes")
        actual_bitsandbytes = str(bitsandbytes.__version__)
        if actual_bitsandbytes != expected["bitsandbytes_version"]:
            reasons.append(
                "bitsandbytes runtime mismatch: "
                f"expected {expected['bitsandbytes_version']}, "
                f"found {actual_bitsandbytes}"
            )

    installed_names = sorted(
        {
            distribution.metadata["Name"].lower()
            for distribution in importlib_metadata.distributions()
            if distribution.metadata.get("Name")
        }
    )
    forbidden_patterns = compatibility.get("forbidden_lock_package_patterns", [])
    forbidden_installed = sorted(
        name
        for name in installed_names
        if any(re.fullmatch(pattern, name) for pattern in forbidden_patterns)
    )
    if forbidden_installed:
        reasons.append(
            "runtime contains an incompatible CUDA stack: " + ", ".join(forbidden_installed)
        )

    return {
        "schema_version": "distillery.training.ml-runtime-check.v1",
        "status": "compatible" if not reasons else "blocked",
        "expected": expected,
        "actual": {
            "python_version": actual_python,
            "torch_version": actual_torch,
            "torch_cuda_version": actual_cuda,
            "cudnn_major": actual_cudnn_major,
            "bitsandbytes_version": actual_bitsandbytes,
            "forbidden_packages": forbidden_installed,
        },
        "reasons": reasons,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    lock_parser = subparsers.add_parser("lock")
    lock_parser.add_argument("--lock", type=Path, required=True)
    lock_parser.add_argument("--compatibility", type=Path, required=True)

    runtime_parser = subparsers.add_parser("runtime")
    runtime_parser.add_argument("--compatibility", type=Path, required=True)
    runtime_parser.add_argument("--require-bitsandbytes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "lock":
            result = check_lock(args.lock, args.compatibility)
        elif args.command == "runtime":
            result = check_runtime(
                args.compatibility,
                require_bitsandbytes=args.require_bitsandbytes,
            )
        else:
            raise AssertionError(f"unhandled command: {args.command}")
    except (ImportError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"compatibility check error: {exc}", file=sys.stderr)
        return BLOCKED_EXIT

    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "compatible":
        for reason in result["reasons"]:
            print(f"compatibility blocker: {reason}", file=sys.stderr)
        return BLOCKED_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
