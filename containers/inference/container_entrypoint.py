#!/usr/bin/env python3
"""PID 1 wrapper for SageMaker inference: ownership drop, signals, uvicorn."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import FrameType

RUNTIME_UID = 1000
RUNTIME_GID = 1000
VERSION_PATH = Path("/opt/distillery/VERSION.json")
MODEL_ROOT = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080

_child: subprocess.Popen[bytes] | None = None
_received_signal: int | None = None


def healthcheck(version_path: Path = VERSION_PATH) -> int:
    payload = json.loads(version_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("VERSION.json must contain an object")
    if payload.get("runtime_uid") != RUNTIME_UID:
        raise ValueError("VERSION.json has unexpected runtime_uid")
    if payload.get("require_offline") is not True:
        raise ValueError("VERSION.json does not require offline serving")
    if payload.get("component") != "inference":
        raise ValueError("VERSION.json component must be inference")
    importlib.import_module("distillery_inference.server")
    return 0


def drop_privileges() -> None:
    if os.geteuid() != 0:
        return
    os.setgid(RUNTIME_GID)
    os.setuid(RUNTIME_UID)


def fix_model_mount_ownership(model_root: Path = MODEL_ROOT) -> None:
    if os.geteuid() != 0:
        return
    if not model_root.exists():
        return
    for path in [model_root, *model_root.rglob("*")]:
        try:
            os.chown(path, RUNTIME_UID, RUNTIME_GID)
        except OSError:
            # Mounted files may be immutable; serving still reads them.
            continue


def handle_signal(signum: int, _frame: FrameType | None) -> None:
    global _received_signal
    _received_signal = signum
    if _child is not None and _child.poll() is None:
        _child.send_signal(signum)


def build_uvicorn_command(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "distillery_inference.server:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "info",
    ]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distillery inference container entrypoint")
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global _child
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.health:
        return healthcheck()

    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "DISTILLERY_INFERENCE_REQUIRE_OFFLINE": "1",
    }.items():
        os.environ.setdefault(key, value)

    fix_model_mount_ownership()
    drop_privileges()

    if os.geteuid() == 0:
        raise SystemExit("refusing to serve inference as root")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    command = build_uvicorn_command(host=args.host, port=args.port)
    _child = subprocess.Popen(command)
    code = _child.wait()
    if _received_signal is not None:
        return 0
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
