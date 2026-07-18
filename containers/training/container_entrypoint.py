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
QWEN72B_DATA_CHANNEL = Path("/opt/ml/input/data/data")
MODEL_OUTPUT_DIR = Path("/opt/ml/model")
CAPABILITY_EVIDENCE_PATH = ("training", "qlora", "capability_evidence")
EXECUTE_ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_TRAINING_EXECUTION"
FOUNDATION_MODE = "foundation"
EMERGENCY_SMOKE_MODE = "emergency-smoke"
QWEN72B_MODE = "qwen72b"
FOUNDATION_TRAINER_MODULE = "distillery.training.entrypoint"
EMERGENCY_TRAINER_MODULE = "experiments.aws_smoke.train"
QWEN72B_LAUNCHER_MODULE = "experiments.qwen72b_fallback.distributed_launcher"
QWEN72B_TRAINER_MODULE = "experiments.qwen72b_fallback.train"

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
    importlib.import_module(FOUNDATION_TRAINER_MODULE)
    importlib.import_module(EMERGENCY_TRAINER_MODULE)
    importlib.import_module(QWEN72B_LAUNCHER_MODULE)
    importlib.import_module(QWEN72B_TRAINER_MODULE)
    return 0


def trainer_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--execution-mode",
        choices=(FOUNDATION_MODE, EMERGENCY_SMOKE_MODE, QWEN72B_MODE),
        default=FOUNDATION_MODE,
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-output-dir", type=Path, default=MODEL_OUTPUT_DIR)
    parser.add_argument(
        "--arm",
        choices=("oracle_sft", "ce_ablation", "logit_kd", "sequence_kd"),
    )
    parser.add_argument("--teacher-responses", type=Path)
    parser.add_argument(
        "--qwen72b-mode",
        choices=("memory_probe", "rehearsal", "full"),
    )
    parser.add_argument("--launch-name")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--memory-probe", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--runtime-image-digest")
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
    if args.execute and args.execution_mode != QWEN72B_MODE:
        if args.execute_acknowledgement != EXECUTE_ACKNOWLEDGEMENT:
            raise ValueError("--execute requires the exact --execute-acknowledgement value")
    elif args.execute_acknowledgement is not None:
        raise ValueError("--execute-acknowledgement is invalid with --validate-only")
    if args.execution_mode == QWEN72B_MODE:
        if not args.execute:
            raise ValueError("qwen72b execution mode requires --execute")
        if args.execute_acknowledgement is not None:
            raise ValueError("qwen72b uses hash-bound typed confirmation, not static ack")
        required = {
            "--qwen72b-mode": args.qwen72b_mode,
            "--launch-name": args.launch_name,
            "--profile": args.profile,
            "--authorization": args.authorization,
            "--models-dir": args.models_dir,
            "--data-dir": args.data_dir,
            "--runtime-image-digest": args.runtime_image_digest,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"qwen72b execution arguments missing: {missing}")
        if args.manifest is not None or args.responses is not None:
            raise ValueError("qwen72b mode forbids foundation manifest/responses arguments")
        if args.arm is not None or args.teacher_responses is not None:
            raise ValueError("qwen72b mode forbids demo arm arguments")
    elif args.execution_mode == EMERGENCY_SMOKE_MODE:
        if not args.execute:
            raise ValueError("emergency-smoke execution mode requires --execute")
        if args.arm is None:
            raise ValueError("emergency-smoke execution mode requires --arm")
    else:
        if args.arm is not None or args.teacher_responses is not None:
            raise ValueError("arm and teacher-responses require emergency-smoke mode")
        if args.manifest is None or args.responses is None:
            raise ValueError("foundation mode requires manifest and responses")
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
    if args.execution_mode == QWEN72B_MODE:
        validate_qwen72b_channels(
            args,
            model_channel=model_channel,
        )
        return
    assert args.manifest is not None
    assert args.responses is not None
    if not args.manifest.is_file():
        raise ValueError(f"manifest channel file missing: {args.manifest}")
    if not args.responses.is_file():
        raise ValueError(f"responses channel file missing: {args.responses}")
    if args.responses.stat().st_size == 0:
        raise ValueError(f"responses channel file is empty: {args.responses}")
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
    if args.execution_mode == EMERGENCY_SMOKE_MODE:
        tags = payload.get("tags")
        if not isinstance(tags, dict):
            raise ValueError("emergency manifest lacks sealed tags")
        if tags.get("RunMode") != "smoke":
            raise ValueError("emergency manifest must seal RunMode=smoke")
        if tags.get("Arm") != args.arm:
            raise ValueError("emergency manifest arm differs from wrapper --arm")
        if tags.get("EnableNetworkIsolation") != "true":
            raise ValueError("emergency manifest must seal network isolation")


def validate_qwen72b_channels(
    args: argparse.Namespace,
    *,
    model_channel: Path,
) -> None:
    from experiments.qwen72b_fallback.profile import (
        Qwen72BTrainingProfile,
        RunKind,
    )
    from experiments.qwen72b_fallback.readiness import (
        ExecutionAction,
        ExecutionAuthorization,
    )

    assert args.profile is not None
    assert args.authorization is not None
    assert args.models_dir is not None
    assert args.data_dir is not None
    assert args.launch_name is not None
    assert args.qwen72b_mode is not None
    required_files = {
        "profile": args.profile,
        "authorization": args.authorization,
        "finance-world evidence": args.data_dir / "finance_world_evidence.json",
        "finance-world records": args.data_dir / "train.jsonl",
    }
    missing = [label for label, path in required_files.items() if not path.is_file()]
    if missing:
        raise ValueError(f"qwen72b input channel files missing: {missing}")
    if not args.models_dir.is_dir():
        raise ValueError("qwen72b models channel directory is missing")
    if args.models_dir != model_channel or args.data_dir != QWEN72B_DATA_CHANNEL:
        raise ValueError("qwen72b paths must equal the sealed SageMaker channel mounts")
    profile = Qwen72BTrainingProfile.model_validate_json(args.profile.read_bytes())
    authorization = ExecutionAuthorization.model_validate_json(args.authorization.read_bytes())
    action = {
        "memory_probe": ExecutionAction.MEMORY_PROBE,
        "rehearsal": ExecutionAction.REHEARSAL,
        "full": ExecutionAction.FULL,
    }[args.qwen72b_mode]
    authorization.require_current(action=action, launch_name=args.launch_name)
    if authorization.evidence_bundle.target_profile_sha256 != profile.profile_sha256:
        raise ValueError("qwen72b profile differs from authorization")
    if args.qwen72b_mode != profile.kind.value and not (
        args.qwen72b_mode == "memory_probe" and profile.kind in {RunKind.REHEARSAL, RunKind.FULL}
    ):
        raise ValueError("qwen72b mode differs from target profile")
    image = authorization.evidence_bundle.ecr_image
    if image is None or args.runtime_image_digest != image.image_digest:
        raise ValueError("qwen72b runtime image digest differs from authorization")
    if args.qwen72b_mode in {"rehearsal", "full"}:
        if args.memory_probe is None or not args.memory_probe.is_file():
            raise ValueError("qwen72b training requires measured memory-probe channel")
        if authorization.evidence_bundle.memory_probe is None:
            raise ValueError("qwen72b authorization lacks measured memory probe")
    elif args.memory_probe is not None:
        raise ValueError("qwen72b memory-probe mode forbids a prior probe file")


def prepare_runtime_directories(
    output_dir: Path,
    *,
    model_output_dir: Path | None = None,
    failure_path: Path = FAILURE_PATH,
    uid: int = RUNTIME_UID,
    gid: int = RUNTIME_GID,
) -> None:
    directories = {output_dir, failure_path.parent}
    if model_output_dir is not None:
        directories.add(model_output_dir)
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


def assert_runtime_writable(
    output_dir: Path,
    failure_path: Path = FAILURE_PATH,
    model_output_dir: Path | None = None,
) -> None:
    directories = [output_dir, failure_path.parent]
    if model_output_dir is not None:
        directories.append(model_output_dir)
    for directory in directories:
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


def build_trainer_command(
    args: argparse.Namespace,
    *,
    model_channel: Path = MODEL_CHANNEL,
    dataset_channel: Path = DATASET_CHANNEL,
) -> list[str]:
    if args.execution_mode == QWEN72B_MODE:
        command = [
            sys.executable,
            "-m",
            QWEN72B_LAUNCHER_MODULE,
            "--mode",
            str(args.qwen72b_mode),
            "--launch-name",
            str(args.launch_name),
            "--profile",
            str(args.profile),
            "--authorization",
            str(args.authorization),
            "--models-dir",
            str(args.models_dir),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(args.output_dir),
            "--runtime-image-digest",
            str(args.runtime_image_digest),
        ]
        if args.memory_probe is not None:
            command.extend(["--memory-probe", str(args.memory_probe)])
        return command
    if args.execution_mode == FOUNDATION_MODE:
        command = [
            sys.executable,
            "-m",
            FOUNDATION_TRAINER_MODULE,
            "--manifest",
            str(args.manifest),
            "--responses",
            str(args.responses),
            "--output-dir",
            str(args.output_dir),
        ]
        if args.execute:
            command.extend(
                [
                    "--execute",
                    "--execute-acknowledgement",
                    str(args.execute_acknowledgement),
                ]
            )
        else:
            command.append("--validate-only")
        return command

    command = [
        sys.executable,
        "-m",
        EMERGENCY_TRAINER_MODULE,
        "--manifest",
        str(args.manifest),
        "--arm",
        str(args.arm),
        "--dataset-dir",
        str(dataset_channel),
        "--models-dir",
        str(model_channel),
        "--output-dir",
        str(args.output_dir),
        "--model-output-dir",
        str(args.model_output_dir),
    ]
    if args.teacher_responses is not None:
        command.extend(["--teacher-responses", str(args.teacher_responses)])
    return command


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
        prepare_runtime_directories(
            args.output_dir,
            model_output_dir=args.model_output_dir,
            failure_path=failure_path,
        )
        drop_runtime_privileges()
        assert_runtime_writable(
            args.output_dir,
            failure_path,
            model_output_dir=args.model_output_dir,
        )
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

    command = build_trainer_command(
        args,
        model_channel=model_channel,
        dataset_channel=dataset_channel,
    )
    print(
        json.dumps(
            {
                "event": "trainer_dispatch",
                "execution_mode": args.execution_mode,
                "trainer_module": command[2],
                "arm": args.arm,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return run_child(command, failure_path=failure_path)


if __name__ == "__main__":
    raise SystemExit(main())
