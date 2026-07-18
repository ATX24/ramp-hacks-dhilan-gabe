"""Spawn eight isolated-GPU trainer children and kill peers on rank death."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from experiments.qwen72b_fallback.profile import (
    Qwen72BTrainingProfile,
    RunKind,
)
from experiments.qwen72b_fallback.readiness import (
    ExecutionAction,
    ExecutionAuthorization,
)

WORLD_SIZE = 8
RANK_DEATH_GRACE_SECONDS = 30


def _available_gpu_ids() -> tuple[str, ...]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        values = tuple(item.strip() for item in visible.split(","))
    else:
        values = tuple(str(index) for index in range(WORLD_SIZE))
    if len(values) != WORLD_SIZE or any(not value.isdecimal() for value in values):
        raise RuntimeError("launcher requires exactly eight visible physical GPU IDs")
    return values


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _action_for_mode(mode: str) -> ExecutionAction:
    return {
        "memory_probe": ExecutionAction.MEMORY_PROBE,
        "rehearsal": ExecutionAction.REHEARSAL,
        "full": ExecutionAction.FULL,
    }[mode]


def _terminate_children(children: list[subprocess.Popen[bytes]]) -> None:
    for child in children:
        if child.poll() is None:
            child.terminate()
    deadline = time.monotonic() + RANK_DEATH_GRACE_SECONDS
    while time.monotonic() < deadline and any(child.poll() is None for child in children):
        time.sleep(0.2)
    for child in children:
        if child.poll() is None:
            child.kill()
    for child in children:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def launch_children(args: argparse.Namespace) -> int:
    profile = Qwen72BTrainingProfile.model_validate_json(args.profile.read_bytes())
    authorization = ExecutionAuthorization.model_validate_json(args.authorization.read_bytes())
    action = _action_for_mode(args.mode)
    authorization.require_current(action=action, launch_name=args.launch_name)
    if args.mode != profile.kind.value and not (
        args.mode == "memory_probe" and profile.kind in {RunKind.REHEARSAL, RunKind.FULL}
    ):
        raise ValueError("launcher mode differs from target profile kind")
    gpu_ids = _available_gpu_ids()
    master_port = _free_loopback_port()
    common_command = [
        sys.executable,
        "-m",
        "experiments.qwen72b_fallback.train",
        "--child",
        "--mode",
        args.mode,
        "--launch-name",
        args.launch_name,
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
        args.runtime_image_digest,
    ]
    if args.memory_probe is not None:
        common_command.extend(["--memory-probe", str(args.memory_probe)])
    children: list[subprocess.Popen[bytes]] = []
    try:
        for rank, gpu_id in enumerate(gpu_ids):
            env = dict(os.environ)
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu_id,
                    "RANK": str(rank),
                    "WORLD_SIZE": str(WORLD_SIZE),
                    "LOCAL_RANK": "0",
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": str(master_port),
                }
            )
            children.append(subprocess.Popen(common_command, env=env))
        deadline = time.monotonic() + profile.max_runtime_seconds
        while time.monotonic() < deadline:
            return_codes = [child.poll() for child in children]
            failed = [code for code in return_codes if code not in {None, 0}]
            if failed:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                (args.output_dir / "launcher-failure.json").write_text(
                    json.dumps(
                        {
                            "return_codes": return_codes,
                            "failed_return_codes": failed,
                            "peer_termination_required": True,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                _terminate_children(children)
                return int(failed[0])
            if all(code == 0 for code in return_codes):
                completion_path = args.output_dir / "completion" / "all-ranks.json"
                completion = json.loads(completion_path.read_bytes())
                if completion.get("acknowledged_ranks") != list(range(WORLD_SIZE)):
                    raise RuntimeError("all-rank completion artifact is incomplete")
                return 0
            time.sleep(0.2)
        _terminate_children(children)
        raise RuntimeError("distributed launcher exceeded sealed runtime")
    finally:
        if any(child.poll() is None for child in children):
            _terminate_children(children)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("memory_probe", "rehearsal", "full"),
        required=True,
    )
    parser.add_argument("--launch-name", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--memory-probe", type=Path)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-image-digest", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return launch_children(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"qwen72b launcher failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(143))
    raise SystemExit(main())
