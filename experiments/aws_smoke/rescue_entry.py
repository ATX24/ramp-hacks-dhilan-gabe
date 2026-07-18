#!/usr/bin/env python3
"""Script Mode rescue entrypoint for one network-isolated oracle_sft smoke job.

Installed into the SageMaker ``code`` channel. Performs an offline local-wheel
install (no network), then dispatches the committed emergency trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
WHEELS_DIR = CODE_ROOT / "wheels"
BUNDLE_MANIFEST = CODE_ROOT / "SOURCE_BUNDLE_MANIFEST.json"
REPO_ROOT = CODE_ROOT
SRC_ROOT = REPO_ROOT / "src"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_bundle() -> str:
    payload = json.loads(BUNDLE_MANIFEST.read_text(encoding="utf-8"))
    files = payload.get("files")
    source_revision = payload.get("source_revision")
    if not isinstance(files, dict) or not isinstance(source_revision, str):
        raise ValueError("invalid sealed source-bundle manifest")
    for relative, expected in files.items():
        path = CODE_ROOT / str(relative)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source-bundle integrity mismatch: {relative}")
    print(
        json.dumps(
            {
                "event": "rescue_source_bundle_verified",
                "file_count": len(files),
                "source_revision": source_revision,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return source_revision


def _install_offline_wheels() -> None:
    if not WHEELS_DIR.is_dir():
        raise FileNotFoundError(f"missing offline wheels directory: {WHEELS_DIR}")
    wheels = sorted(WHEELS_DIR.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError(f"no wheels present under {WHEELS_DIR}")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        f"--find-links={WHEELS_DIR}",
        "--disable-pip-version-check",
        "--no-cache-dir",
        *[str(path) for path in wheels],
    ]
    print(
        json.dumps(
            {
                "event": "rescue_offline_wheel_install",
                "wheel_count": len(wheels),
                "wheels": [path.name for path in wheels],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    subprocess.check_call(command)


def _prepare_pythonpath() -> None:
    for path in (str(REPO_ROOT), str(SRC_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(REPO_ROOT), str(SRC_ROOT)]
    if existing:
        parts.append(existing)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)


def _smoke_imports() -> None:
    import experiments.aws_smoke.train as train  # noqa: F401
    import torch  # noqa: F401
    from peft import LoraConfig  # noqa: F401
    from transformers import AutoModelForCausalLM  # noqa: F401

    print(
        json.dumps(
            {
                "event": "rescue_import_smoke_ok",
                "trainer_module": "experiments.aws_smoke.train",
                "torch_version": getattr(torch, "__version__", "unknown"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _normalize_model_channel(manifest_path: Path, models_dir: Path) -> Path:
    """Restore the trainer's org/name/revision tree from an exact S3 prefix."""
    if not (models_dir / "config.json").is_file():
        return models_dir
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    student = payload["models"]["student"]
    model_id = str(student["id"])
    revision = str(student["revision"])
    org, separator, name = model_id.partition("/")
    if not separator or "/" in name or len(revision) != 40:
        raise ValueError("invalid sealed student model identity")
    root = Path("/tmp/distillery-rescue-models")
    shutil.rmtree(root, ignore_errors=True)
    destination = root / org / name / revision
    destination.mkdir(parents=True)
    for source in sorted(models_dir.iterdir()):
        if source.is_file():
            (destination / source.name).symlink_to(source)
    if not (destination / "config.json").is_file():
        raise FileNotFoundError("normalized student model channel lacks config.json")
    return root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rescue_entry")
    parser.add_argument("--arm", required=True, choices=["oracle_sft"])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/opt/ml/input/data/manifest/manifest.json"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/opt/ml/input/data/dataset"),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("/opt/ml/input/data/models"),
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("/opt/ml/input/data/responses/responses.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/opt/ml/output/data"),
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=Path("/opt/ml/model"),
    )
    parser.add_argument(
        "--import-smoke-only",
        action="store_true",
        help="Validate imports/paths then exit without training",
    )
    parser.add_argument(
        "--skip-wheel-install",
        action="store_true",
        help="Only valid with --import-smoke-only for local staged validation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _verify_source_bundle()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest missing: {args.manifest}")
    if not args.responses.is_file() or args.responses.stat().st_size == 0:
        raise FileNotFoundError(f"responses channel missing/empty: {args.responses}")
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset channel missing: {args.dataset_dir}")
    if not args.models_dir.is_dir():
        raise FileNotFoundError(f"models channel missing: {args.models_dir}")
    if not (REPO_ROOT / "experiments" / "aws_smoke" / "train.py").is_file():
        raise FileNotFoundError("emergency trainer missing from sealed source bundle")
    if args.skip_wheel_install and not args.import_smoke_only:
        raise ValueError("--skip-wheel-install is only valid with --import-smoke-only")

    if not args.skip_wheel_install:
        _install_offline_wheels()
    _prepare_pythonpath()
    _smoke_imports()
    if args.import_smoke_only:
        return 0

    from experiments.aws_smoke.train import main as train_main

    normalized_models_dir = _normalize_model_channel(args.manifest, args.models_dir)
    return train_main(
        [
            "--manifest",
            str(args.manifest),
            "--arm",
            args.arm,
            "--dataset-dir",
            str(args.dataset_dir),
            "--models-dir",
            str(normalized_models_dir),
            "--output-dir",
            str(args.output_dir),
            "--model-output-dir",
            str(args.model_output_dir),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
