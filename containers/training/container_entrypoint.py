#!/usr/bin/env python3
"""PID 1 wrapper for SageMaker output ownership, signals, and failures."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from types import FrameType
from typing import Any

RUNTIME_UID = 1000
RUNTIME_GID = 1000
FAILURE_PATH = Path("/opt/ml/output/failure")
VERSION_PATH = Path("/opt/distillery/VERSION.json")
MODEL_CHANNEL = Path("/opt/ml/input/data/models")
DATASET_CHANNEL = Path("/opt/ml/input/data/dataset")
CAPABILITY_EVIDENCE_PATH = ("training", "qlora", "capability_evidence")
EXECUTE_ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_TRAINING_EXECUTION"

_child: subprocess.Popen[bytes] | None = None
_received_signal: int | None = None


def healthcheck(version_path: Path = VERSION_PATH) -> int:
    payload = json.loads(version_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("VERSION.json must contain an object")
    if payload.get("runtime_uid") != RUNTIME_UID:
        raise ValueError("VERSION.json has unexpected runtime_uid")
    if payload.get("require_pinned_revision") is not True:
        raise ValueError("VERSION.json does not require pinned model revisions")
    importlib.import_module("distillery.training.entrypoint")
    return 0


def trainer_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-acknowledgement")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def validate_trainer_arguments(argv: list[str]) -> argparse.Namespace:
    args, unknown = trainer_argument_parser().parse_known_args(argv)
    if unknown:
        raise ValueError(f"unexpected trainer arguments: {unknown}")
    if args.execute == args.validate_only:
        raise ValueError("exactly one of --execute or --validate-only is required")
    if args.execute:
        if args.execute_acknowledgement != EXECUTE_ACKNOWLEDGEMENT:
            raise ValueError("--execute requires the exact --execute-acknowledgement value")
    elif args.execute_acknowledgement is not None:
        raise ValueError("--execute-acknowledgement is invalid with --validate-only")
    return args


def nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for component in path:
        if not isinstance(value, dict) or component not in value:
            joined = ".".join(path)
            raise ValueError(f"manifest lacks required embedded {joined}")
        value = value[component]
    return value


def validate_input_channels(
    args: argparse.Namespace,
    *,
    model_channel: Path = MODEL_CHANNEL,
    dataset_channel: Path = DATASET_CHANNEL,
) -> None:
    if not args.manifest.is_file():
        raise ValueError(f"manifest channel file missing: {args.manifest}")
    if not args.responses.is_file():
        raise ValueError(f"responses channel file missing: {args.responses}")
    if not model_channel.is_dir():
        raise ValueError(f"models channel directory missing: {model_channel}")
    if not dataset_channel.is_dir():
        raise ValueError(f"dataset channel directory missing: {dataset_channel}")

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    evidence = nested_value(payload, CAPABILITY_EVIDENCE_PATH)
    if not isinstance(evidence, dict):
        raise ValueError("embedded capability evidence must be a JSON object")


def prepare_runtime_directories(
    output_dir: Path,
    *,
    failure_path: Path = FAILURE_PATH,
    uid: int = RUNTIME_UID,
    gid: int = RUNTIME_GID,
) -> None:
    directories = {output_dir, failure_path.parent}
    model_root = Path("/opt/ml/model")
    if output_dir.is_relative_to(model_root):
        directories.add(model_root)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        if os.geteuid() == 0:
            os.chown(directory, uid, gid)
            os.chmod(directory, 0o750)


def drop_runtime_privileges(
    *,
    uid: int = RUNTIME_UID,
    gid: int = RUNTIME_GID,
) -> None:
    if os.geteuid() != 0:
        return
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)
    if os.geteuid() != uid or os.getegid() != gid:
        raise RuntimeError("failed to drop root privileges")


def assert_runtime_writable(output_dir: Path, failure_path: Path = FAILURE_PATH) -> None:
    for directory in (output_dir, failure_path.parent):
        if not os.access(directory, os.W_OK | os.X_OK):
            raise PermissionError(
                f"runtime uid {os.geteuid()} cannot write SageMaker path {directory}"
            )


def atomic_write_failure(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_message = " ".join(message.splitlines())[:1024] + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(safe_message)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def forward_signal(signum: int, _frame: FrameType | None) -> None:
    global _received_signal
    _received_signal = signum
    if _child is not None and _child.poll() is None:
        os.killpg(_child.pid, signum)


def run_child(command: list[str], *, failure_path: Path = FAILURE_PATH) -> int:
    global _child, _received_signal
    _received_signal = None
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, forward_signal)

    _child = subprocess.Popen(command, start_new_session=True)
    if _received_signal is not None:
        os.killpg(_child.pid, _received_signal)
    try:
        return_code = _child.wait()
    finally:
        _child = None

    normalized_return_code = 128 - return_code if return_code < 0 else return_code
    if normalized_return_code != 0:
        signal_context = f"; forwarded signal {_received_signal}" if _received_signal else ""
        atomic_write_failure(
            failure_path,
            f"Distillery trainer exited with status {normalized_return_code}{signal_context}",
        )
    return normalized_return_code


def main(argv: list[str] | None = None) -> int:
    trainer_argv = list(sys.argv[1:] if argv is None else argv)
    if trainer_argv == ["--health"]:
        try:
            return healthcheck()
        except (ImportError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"container healthcheck failed: {exc}", file=sys.stderr)
            return 1
    failure_path = Path(os.environ.get("DISTILLERY_FAILURE_PATH", FAILURE_PATH))
    model_channel = Path(os.environ.get("DISTILLERY_SAGEMAKER_MODEL_INPUT", MODEL_CHANNEL))
    dataset_channel = Path(os.environ.get("DISTILLERY_SAGEMAKER_DATASET_INPUT", DATASET_CHANNEL))
    try:
        args = validate_trainer_arguments(trainer_argv)
        prepare_runtime_directories(args.output_dir, failure_path=failure_path)
        drop_runtime_privileges()
        assert_runtime_writable(args.output_dir, failure_path)
        validate_input_channels(
            args,
            model_channel=model_channel,
            dataset_channel=dataset_channel,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        try:
            atomic_write_failure(failure_path, f"Container preflight failed: {exc}")
        except OSError:
            pass
        print(f"container preflight failed: {exc}", file=sys.stderr)
        return 1

    command = [sys.executable, "-m", "distillery.training.entrypoint", *trainer_argv]
    return run_child(command, failure_path=failure_path)


if __name__ == "__main__":
    raise SystemExit(main())
