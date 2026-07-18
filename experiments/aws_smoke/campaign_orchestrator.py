"""Fail-closed independent-per-GPU orchestration for supported 4/8-GPU nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, TextIO

from distillery.contracts.hashing import canonical_json_bytes
from experiments.aws_smoke.artifacts import verify_emergency_artifacts
from experiments.aws_smoke.campaign_index import (
    MAX_GPU_COUNT,
    CampaignArmBinding,
    VerifiedCampaignBundle,
    verify_campaign_bundle,
)

TRAINER_MODULE = "experiments.aws_smoke.train"
SINGLE_GPU_ENV = "DISTILLERY_CAMPAIGN_SINGLE_GPU"
SHARED_CAMPAIGN_ENV = "DISTILLERY_SHARED_CAMPAIGN"
DEFAULT_OUTPUT_ROOT = Path("/opt/ml/output/data")
DEFAULT_MODEL_ROOT = Path("/opt/ml/model")
DEFAULT_RUNTIME_ROOT = Path("/tmp/distillery-campaign")
DEFAULT_DATASET_DIR = Path("/opt/ml/input/data/dataset")
DEFAULT_MODELS_DIR = Path("/opt/ml/input/data/models")
DEFAULT_SHUTDOWN_SECONDS = 20.0
_NANOSECONDS_PER_HOUR = 3_600_000_000_000

_PASSTHROUGH_ENV = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
        "PYTHONPATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TZ",
    }
)
_PATH_ENV = frozenset(
    {
        "HOME",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "PATH",
        "PYTHONPATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
    }
)


class ProcessLike(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


PopenFactory = Callable[..., ProcessLike]
ArtifactVerifier = Callable[[Path], dict[str, Any]]
ProcessGroupSignaler = Callable[[int, int], None]
GpuCountProvider = Callable[[], int]


@dataclass(slots=True)
class _ChildRuntime:
    binding: CampaignArmBinding
    manifest_path: Path
    output_dir: Path
    model_dir: Path
    work_dir: Path
    failure_path: Path
    stdout_path: Path
    stderr_path: Path
    process: ProcessLike | None = None
    stdout_handle: TextIO | None = None
    stderr_handle: TextIO | None = None
    status: str = "not_started"
    return_code: int | None = None
    error: str | None = None
    artifacts_verified: bool = False


@dataclass(frozen=True, slots=True)
class CampaignExecutionResult:
    status: str
    summary_path: Path
    inventory_path: Path
    parent_cost_microusd: int


def _safe_path_list(value: str, *, field_name: str) -> str:
    for component in value.split(os.pathsep):
        if not component or not Path(component).is_absolute():
            raise ValueError(f"{field_name} contains an empty or non-absolute path component")
    return value


def build_child_environment(
    parent: Mapping[str, str],
    *,
    gpu_slot: int,
    seed: int,
    failure_path: Path,
) -> dict[str, str]:
    """Build a minimal credential-free deterministic environment for one child."""
    if not 0 <= gpu_slot < MAX_GPU_COUNT:
        raise ValueError(f"gpu_slot must be between zero and {MAX_GPU_COUNT - 1}")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must fit PYTHONHASHSEED's unsigned 32-bit range")
    if not failure_path.is_absolute():
        raise ValueError("failure_path must be absolute")
    child: dict[str, str] = {}
    for key in sorted(_PASSTHROUGH_ENV):
        value = parent.get(key)
        if value is None:
            continue
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError(f"unsafe control character in environment variable {key}")
        child[key] = _safe_path_list(value, field_name=key) if key in _PATH_ENV else value

    child.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_VISIBLE_DEVICES": str(gpu_slot),
            "DISTILLERY_FAILURE_PATH": str(failure_path),
            SHARED_CAMPAIGN_ENV: "1",
            SINGLE_GPU_ENV: "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONHASHSEED": str(seed),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return child


def parent_cost_microusd(*, hourly_price_microusd: int, elapsed_ns: int) -> int:
    """Ceiling-rounded integer charge for one instance and one wall-clock span."""
    if hourly_price_microusd <= 0:
        raise ValueError("hourly_price_microusd must be positive")
    if elapsed_ns < 0:
        raise ValueError("elapsed_ns must be nonnegative")
    numerator = hourly_price_microusd * elapsed_ns
    return (numerator + _NANOSECONDS_PER_HOUR - 1) // _NANOSECONDS_PER_HOUR


def allocate_parent_cost(
    parent_cost: int,
    bindings: Sequence[CampaignArmBinding],
) -> tuple[int, ...]:
    """Equal-share micro-USD allocation with deterministic ordered remainder."""
    if parent_cost < 0:
        raise ValueError("parent_cost must be nonnegative")
    if not bindings:
        raise ValueError("cannot allocate cost without campaign arms")
    quotient, remainder = divmod(parent_cost, len(bindings))
    allocations = tuple(
        quotient + (1 if index < remainder else 0) for index, _ in enumerate(bindings)
    )
    if sum(allocations) != parent_cost:
        raise RuntimeError("internal cost allocation error")
    return allocations


def _reject_symlink_ancestors(path: Path, *, field_name: str) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"{field_name} must not traverse a symlink")
        if current == current.parent:
            return
        current = current.parent


def _validate_absolute_directory(path: Path, *, field_name: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    _reject_symlink_ancestors(path, field_name=field_name)
    if not path.is_dir():
        raise FileNotFoundError(f"{field_name} must be a regular directory: {path}")
    return path.resolve(strict=True)


def _validate_output_base(path: Path, *, field_name: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    _reject_symlink_ancestors(path, field_name=field_name)
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{field_name} must resolve to a regular directory")
    return path.resolve(strict=True)


def _require_disjoint_roots(named_roots: Mapping[str, Path]) -> None:
    items = list(named_roots.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(f"{left_name} and {right_name} must be disjoint directories")


def _validate_interpreter(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("trainer interpreter must be an absolute path")
    if not path.exists() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("trainer interpreter must be an executable file")
    return path


def _detected_gpu_count() -> int:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to verify campaign GPU topology") from exc
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_canonical_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(payload)))


def _write_integrity_inventory(
    campaign_output: Path,
    campaign_model: Path,
) -> Path:
    inventory_path = campaign_output / "campaign" / "integrity.json"
    files: list[dict[str, Any]] = []
    for namespace, root in (("output", campaign_output), ("model", campaign_model)):
        for path in sorted(root.rglob("*")):
            if path == inventory_path:
                continue
            if path.is_symlink():
                raise ValueError(f"campaign {namespace} contains a symlink: {path}")
            if not path.is_file():
                continue
            files.append(
                {
                    "path": (
                        PurePosixPath(namespace) / path.relative_to(root).as_posix()
                    ).as_posix(),
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    _write_canonical_json(
        inventory_path,
        {
            "schema_version": "distillery.aws_smoke.campaign_integrity.v1",
            "files": files,
        },
    )
    return inventory_path


def _signal_process_group(
    process: ProcessLike,
    signum: int,
    *,
    signaler: ProcessGroupSignaler,
) -> None:
    if process.poll() is not None:
        return
    try:
        signaler(process.pid, signum)
    except ProcessLookupError:
        return


def _terminate_running(
    children: Sequence[_ChildRuntime],
    *,
    shutdown_seconds: float,
    signaler: ProcessGroupSignaler,
    monotonic: Callable[[], float],
) -> None:
    running = [
        child for child in children if child.process is not None and child.process.poll() is None
    ]
    for child in running:
        _signal_process_group(child.process, signal.SIGTERM, signaler=signaler)
        if child.status == "running":
            child.status = "terminating"

    deadline = monotonic() + shutdown_seconds
    for child in running:
        process = child.process
        assert process is not None
        remaining = max(0.0, deadline - monotonic())
        try:
            child.return_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _signal_process_group(process, signal.SIGKILL, signaler=signaler)
            try:
                child.return_code = process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                child.return_code = None
        if child.status == "terminating":
            child.status = "terminated"


def _close_child_logs(children: Sequence[_ChildRuntime]) -> None:
    for child in children:
        for handle in (child.stdout_handle, child.stderr_handle):
            if handle is not None and not handle.closed:
                handle.flush()
                handle.close()


def _prepare_children(
    bundle: VerifiedCampaignBundle,
    *,
    campaign_output: Path,
    campaign_model: Path,
    campaign_runtime: Path,
) -> list[_ChildRuntime]:
    children: list[_ChildRuntime] = []
    for binding in bundle.index.arms:
        source = bundle.root.joinpath(*Path(binding.manifest_path).parts)
        work_dir = campaign_runtime / binding.run_id
        input_dir = work_dir / "input"
        output_dir = campaign_output / "arms" / binding.run_id
        model_dir = campaign_model / binding.run_id
        log_dir = campaign_output / "logs" / binding.run_id
        for directory in (input_dir, output_dir, model_dir, log_dir):
            directory.mkdir(parents=True, exist_ok=False)
        isolated_manifest = input_dir / "manifest.json"
        sealed_manifest = output_dir / "sealed_manifest.json"
        source_bytes = source.read_bytes()
        isolated_manifest.write_bytes(source_bytes)
        sealed_manifest.write_bytes(source_bytes)
        if _sha256_file(isolated_manifest) != _sha256_file(source) or _sha256_file(
            sealed_manifest
        ) != _sha256_file(source):
            raise RuntimeError(f"{binding.run_id}: isolated manifest copy changed")
        children.append(
            _ChildRuntime(
                binding=binding,
                manifest_path=isolated_manifest,
                output_dir=output_dir,
                model_dir=model_dir,
                work_dir=work_dir,
                failure_path=work_dir / "failure.txt",
                stdout_path=log_dir / "stdout.log",
                stderr_path=log_dir / "stderr.log",
            )
        )
    return children


def _child_command(
    child: _ChildRuntime,
    *,
    interpreter: Path,
    dataset_dir: Path,
    models_dir: Path,
) -> list[str]:
    return [
        str(interpreter),
        "-m",
        TRAINER_MODULE,
        "--manifest",
        str(child.manifest_path),
        "--arm",
        child.binding.arm,
        "--dataset-dir",
        str(dataset_dir),
        "--models-dir",
        str(models_dir),
        "--output-dir",
        str(child.output_dir),
        "--model-output-dir",
        str(child.model_dir),
    ]


def run_campaign(
    *,
    campaign_root: Path,
    expected_index_sha256: str,
    dataset_dir: Path,
    models_dir: Path,
    output_root: Path,
    model_root: Path,
    runtime_root: Path,
    interpreter: Path,
    timeout_seconds: int | None = None,
    shutdown_seconds: float = DEFAULT_SHUTDOWN_SECONDS,
    parent_env: Mapping[str, str] | None = None,
    popen_factory: PopenFactory = subprocess.Popen,
    artifact_verifier: ArtifactVerifier = verify_emergency_artifacts,
    process_group_signaler: ProcessGroupSignaler = os.killpg,
    gpu_count_provider: GpuCountProvider = _detected_gpu_count,
    monotonic: Callable[[], float] = time.monotonic,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> CampaignExecutionResult:
    """Run all sealed arms concurrently; any failure terminates the remaining arms."""
    started_ns = monotonic_ns()
    started_monotonic = monotonic()
    bundle = verify_campaign_bundle(
        campaign_root,
        expected_index_sha256=expected_index_sha256,
    )
    detected_gpu_count = gpu_count_provider()
    if detected_gpu_count != bundle.index.hardware.gpu_count:
        raise RuntimeError(
            "runtime GPU count does not match sealed hardware profile: "
            f"expected={bundle.index.hardware.gpu_count} actual={detected_gpu_count}"
        )
    interpreter = _validate_interpreter(interpreter)
    dataset_dir = _validate_absolute_directory(dataset_dir, field_name="dataset_dir")
    models_dir = _validate_absolute_directory(models_dir, field_name="models_dir")
    output_root = _validate_output_base(output_root, field_name="output_root")
    model_root = _validate_output_base(model_root, field_name="model_root")
    runtime_root = _validate_output_base(runtime_root, field_name="runtime_root")
    _require_disjoint_roots(
        {
            "campaign_root": bundle.root,
            "dataset_dir": dataset_dir,
            "models_dir": models_dir,
            "output_root": output_root,
            "model_root": model_root,
            "runtime_root": runtime_root,
        }
    )
    sealed_timeout = bundle.index.max_runtime_seconds
    if timeout_seconds is not None and timeout_seconds != sealed_timeout:
        raise ValueError("timeout_seconds must equal the sealed campaign runtime")
    timeout_seconds = sealed_timeout
    if shutdown_seconds <= 0:
        raise ValueError("shutdown_seconds must be positive")

    campaign_output = output_root / bundle.index.campaign_id
    campaign_model = model_root / bundle.index.campaign_id
    campaign_runtime = runtime_root / bundle.index.campaign_id
    for path in (campaign_output, campaign_model, campaign_runtime):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"campaign path collision: {path}")
    for path in (campaign_output, campaign_model, campaign_runtime):
        path.mkdir(parents=True, exist_ok=False)

    children = _prepare_children(
        bundle,
        campaign_output=campaign_output,
        campaign_model=campaign_model,
        campaign_runtime=campaign_runtime,
    )
    deadline = started_monotonic + timeout_seconds
    failure_reason: str | None = None
    parent_status = "running"
    environment = dict(os.environ if parent_env is None else parent_env)

    try:
        for child, manifest in zip(children, bundle.manifests, strict=True):
            child.stdout_handle = child.stdout_path.open("w", encoding="utf-8")
            child.stderr_handle = child.stderr_path.open("w", encoding="utf-8")
            command = _child_command(
                child,
                interpreter=interpreter,
                dataset_dir=dataset_dir,
                models_dir=models_dir,
            )
            child_env = build_child_environment(
                environment,
                gpu_slot=child.binding.gpu_slot,
                seed=manifest.training.seed,
                failure_path=child.failure_path,
            )
            try:
                child.process = popen_factory(
                    command,
                    cwd=child.work_dir,
                    env=child_env,
                    stdin=subprocess.DEVNULL,
                    stdout=child.stdout_handle,
                    stderr=child.stderr_handle,
                    start_new_session=True,
                    close_fds=True,
                )
            except OSError as exc:
                child.status = "startup_failed"
                child.error = f"{type(exc).__name__}: {exc}"
                failure_reason = f"{child.binding.run_id}: child startup failed"
                break
            child.status = "running"
            immediate = child.process.poll()
            if immediate is not None:
                child.return_code = immediate
                child.status = "succeeded" if immediate == 0 else "failed"
                if immediate != 0:
                    failure_reason = (
                        f"{child.binding.run_id}: child exited {immediate} during startup"
                    )
                    break

        if failure_reason is None:
            while True:
                if monotonic() >= deadline:
                    failure_reason = "campaign exceeded sealed runtime"
                    parent_status = "timed_out"
                    break
                all_finished = True
                for child in children:
                    if child.process is None or child.status != "running":
                        continue
                    return_code = child.process.poll()
                    if return_code is None:
                        all_finished = False
                        continue
                    child.return_code = return_code
                    child.status = "succeeded" if return_code == 0 else "failed"
                    if return_code != 0:
                        failure_reason = f"{child.binding.run_id}: child exited {return_code}"
                        break
                if failure_reason is not None or all_finished:
                    break
                sleep(0.05)

        if failure_reason is not None:
            _terminate_running(
                children,
                shutdown_seconds=shutdown_seconds,
                signaler=process_group_signaler,
                monotonic=monotonic,
            )
            if parent_status != "timed_out":
                parent_status = "failed"
        else:
            parent_status = "succeeded"
    except BaseException as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        parent_status = "failed"
        _terminate_running(
            children,
            shutdown_seconds=shutdown_seconds,
            signaler=process_group_signaler,
            monotonic=monotonic,
        )
    finally:
        _close_child_logs(children)

    if parent_status == "succeeded":
        for child in children:
            try:
                artifact_verifier(child.output_dir)
            except BaseException as exc:
                child.status = "artifact_failed"
                child.error = f"{type(exc).__name__}: {exc}"
                parent_status = "failed"
                failure_reason = f"{child.binding.run_id}: artifact verification failed"
                break
            child.artifacts_verified = True

    ended_ns = monotonic_ns()
    elapsed_ns = max(0, ended_ns - started_ns)
    parent_cost = parent_cost_microusd(
        hourly_price_microusd=bundle.index.pricing.hourly_price_microusd,
        elapsed_ns=elapsed_ns,
    )
    allocations = allocate_parent_cost(parent_cost, bundle.index.arms)
    allocation_records: list[dict[str, Any]] = []
    for child, allocation in zip(children, allocations, strict=True):
        record = {
            "schema_version": "distillery.aws_smoke.shared_allocation.v1",
            "run_id": child.binding.run_id,
            "gpu_slot": child.binding.gpu_slot,
            "allocated_cost_microusd": allocation,
            "currency": "USD",
            "allocation_basis": "equal_share_ordered_micro_usd_remainder",
            "parent_campaign_index_sha256": bundle.index_sha256,
        }
        allocation_records.append(record)
        _write_canonical_json(
            campaign_output / "campaign" / "allocations" / f"{child.binding.run_id}.json",
            record,
        )

    child_records = [
        {
            "ordinal": child.binding.ordinal,
            "arm": child.binding.arm,
            "run_id": child.binding.run_id,
            "gpu_slot": child.binding.gpu_slot,
            "manifest_sha256": child.binding.manifest_sha256,
            "protocol_sha256": child.binding.protocol_sha256,
            "status": child.status,
            "return_code": child.return_code,
            "error": child.error,
            "artifacts_verified": child.artifacts_verified,
            "output_path": f"arms/{child.binding.run_id}",
            "model_path": child.binding.run_id,
            "stdout_path": f"logs/{child.binding.run_id}/stdout.log",
            "stderr_path": f"logs/{child.binding.run_id}/stderr.log",
            "allocated_cost_microusd": allocations[index],
        }
        for index, child in enumerate(children)
    ]
    summary = {
        "schema_version": "distillery.aws_smoke.campaign_summary.v1",
        "campaign_id": bundle.index.campaign_id,
        "campaign_index_sha256": bundle.index_sha256,
        "campaign_protocol_sha256": bundle.index.protocol_sha256,
        "status": parent_status,
        "failure_reason": failure_reason,
        "hardware_profile": bundle.index.hardware.model_dump(mode="json"),
        "pricing_evidence": bundle.index.pricing.model_dump(mode="json"),
        "elapsed_nanoseconds": elapsed_ns,
        "parent_cost_microusd": parent_cost,
        "cost_scope": "one_instance_wall_clock",
        "allocation_total_microusd": sum(allocations),
        "allocations": allocation_records,
        "children": child_records,
    }
    if summary["allocation_total_microusd"] != summary["parent_cost_microusd"]:
        raise RuntimeError("campaign allocation does not equal parent cost")
    summary_path = campaign_output / "campaign" / "summary.json"
    _write_canonical_json(summary_path, summary)
    inventory_path = _write_integrity_inventory(campaign_output, campaign_model)
    shutil.rmtree(campaign_runtime, ignore_errors=True)
    return CampaignExecutionResult(
        status=parent_status,
        summary_path=summary_path,
        inventory_path=inventory_path,
        parent_cost_microusd=parent_cost,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="distillery-campaign-orchestrator")
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--expected-index-sha256", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout-seconds", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output = stdout or sys.stdout
    try:
        result = run_campaign(
            campaign_root=args.campaign_root,
            expected_index_sha256=args.expected_index_sha256,
            dataset_dir=args.dataset_dir,
            models_dir=args.models_dir,
            output_root=args.output_root,
            model_root=args.model_root,
            runtime_root=args.runtime_root,
            interpreter=args.python_executable,
            timeout_seconds=args.timeout_seconds,
        )
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=output,
        )
        return 1
    print(
        json.dumps(
            {
                "ok": result.status == "succeeded",
                "status": result.status,
                "summary_path": str(result.summary_path),
                "inventory_path": str(result.inventory_path),
            },
            sort_keys=True,
        ),
        file=output,
    )
    return 0 if result.status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
