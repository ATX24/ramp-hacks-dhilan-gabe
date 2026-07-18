"""Entrypoint argv, channels, privilege drop, signal, and failure tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import DOWNSTREAM_CONTRACT, ENTRYPOINT, load_module

from distillery.training.entrypoint import (
    EXECUTE_ACKNOWLEDGEMENT,
    build_arg_parser,
)


@pytest.fixture(scope="module")
def entrypoint_module():
    return load_module(ENTRYPOINT, "distillery_container_entrypoint")


def test_downstream_argv_matches_actual_trainer_parser() -> None:
    contract = json.loads(DOWNSTREAM_CONTRACT.read_text(encoding="utf-8"))
    parser = build_arg_parser()
    backend = contract["sagemaker_backend"]

    validate_argv = backend["validate_invocation"]["ContainerArguments"]
    validate = parser.parse_args(validate_argv)
    assert validate.manifest == Path("/opt/ml/input/data/manifest/manifest.json")
    assert validate.responses == Path("/opt/ml/input/data/responses/responses.jsonl")
    assert validate.output_dir == Path("/opt/ml/model/validation")
    assert validate.validate_only is True
    assert validate.execute is False

    execute_argv = backend["execute_invocation"]["ContainerArguments"]
    execute = parser.parse_args(execute_argv)
    assert execute.responses == Path("/opt/ml/input/data/responses/responses.jsonl")
    assert execute.output_dir == Path("/opt/ml/model")
    assert execute.execute is True
    assert execute.execute_acknowledgement == EXECUTE_ACKNOWLEDGEMENT

    parser_options = {option for action in parser._actions for option in action.option_strings}
    assert "--capability-evidence" not in parser_options
    evidence = backend["capability_evidence"]
    assert evidence["transport"] == "embedded_manifest"
    assert evidence["json_path"] == "training.qlora.capability_evidence"


def test_emergency_mode_dispatches_real_optimizer_trainer(entrypoint_module) -> None:
    contract = json.loads(DOWNSTREAM_CONTRACT.read_text(encoding="utf-8"))
    emergency = contract["sagemaker_backend"]["emergency_smoke_invocation"]
    args = entrypoint_module.validate_trainer_arguments(emergency["ContainerArguments"])
    command = entrypoint_module.build_trainer_command(
        args,
        model_channel=Path("/opt/ml/input/data/models"),
        dataset_channel=Path("/opt/ml/input/data/dataset"),
    )
    assert args.execution_mode == "emergency-smoke"
    assert command[1:3] == ["-m", "experiments.aws_smoke.train"]
    assert "--dataset-dir" in command
    assert "--models-dir" in command
    assert "--model-output-dir" in command
    assert "--validate-only" not in command
    assert emergency["optimizer_path"] == "torch.optim.AdamW"


def test_wrapper_rejects_missing_or_ambiguous_mode(entrypoint_module) -> None:
    common = [
        "--manifest",
        "/tmp/manifest.json",
        "--responses",
        "/tmp/responses.jsonl",
        "--output-dir",
        "/tmp/output",
    ]
    with pytest.raises(ValueError, match="exactly one"):
        entrypoint_module.validate_trainer_arguments(common)
    with pytest.raises(ValueError, match="exactly one"):
        entrypoint_module.validate_trainer_arguments([*common, "--execute", "--validate-only"])
    with pytest.raises(ValueError, match="exact"):
        entrypoint_module.validate_trainer_arguments(
            [
                *common,
                "--execute",
                "--execute-acknowledgement",
                "wrong",
            ]
        )


def test_channels_and_embedded_capability_evidence_are_required(
    tmp_path: Path,
    entrypoint_module,
) -> None:
    manifest = tmp_path / "manifest.json"
    responses = tmp_path / "responses.jsonl"
    models = tmp_path / "models"
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    responses.write_text("{}\n", encoding="utf-8")
    models.mkdir()
    dataset.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "training": {
                    "qlora": {
                        "capability_evidence": {
                            "schema_version": "distillery.training_capabilities.v1"
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    args = entrypoint_module.validate_trainer_arguments(
        [
            "--manifest",
            str(manifest),
            "--responses",
            str(responses),
            "--output-dir",
            str(output),
            "--validate-only",
        ]
    )
    entrypoint_module.validate_input_channels(
        args,
        model_channel=models,
        dataset_channel=dataset,
    )

    manifest.write_text('{"training":{"qlora":{}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="capability_evidence"):
        entrypoint_module.validate_input_channels(
            args,
            model_channel=models,
            dataset_channel=dataset,
        )


def test_runtime_directories_are_writable_and_failures_atomic(
    tmp_path: Path,
    entrypoint_module,
) -> None:
    output = tmp_path / "model" / "validation"
    failure = tmp_path / "output" / "failure"
    entrypoint_module.prepare_runtime_directories(
        output,
        failure_path=failure,
    )
    entrypoint_module.assert_runtime_writable(output, failure)
    entrypoint_module.atomic_write_failure(
        failure,
        "line one\nline two",
    )
    assert failure.read_text(encoding="utf-8") == "line one line two\n"
    assert list(failure.parent.glob(".failure.*")) == []


def test_healthcheck_validates_version_metadata(
    tmp_path: Path,
    monkeypatch,
    entrypoint_module,
) -> None:
    version = tmp_path / "VERSION.json"
    version.write_text(
        json.dumps(
            {
                "runtime_uid": 1000,
                "require_pinned_revision": True,
            }
        ),
        encoding="utf-8",
    )
    imported: list[str] = []
    monkeypatch.setattr(
        entrypoint_module.importlib,
        "import_module",
        lambda name: imported.append(name),
    )
    assert entrypoint_module.healthcheck(version) == 0
    assert imported == [
        "distillery.training.entrypoint",
        "experiments.aws_smoke.train",
        "experiments.qwen72b_fallback.distributed_launcher",
        "experiments.qwen72b_fallback.train",
    ]

    version.write_text('{"runtime_uid":0}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        entrypoint_module.healthcheck(version)


def test_root_init_drops_groups_gid_and_uid(monkeypatch, entrypoint_module) -> None:
    calls: list[tuple[str, object]] = []
    euid_values = iter((0, 1000))
    monkeypatch.setattr(entrypoint_module.os, "geteuid", lambda: next(euid_values))
    monkeypatch.setattr(entrypoint_module.os, "getegid", lambda: 1000)
    monkeypatch.setattr(
        entrypoint_module.os,
        "setgroups",
        lambda groups: calls.append(("groups", groups)),
    )
    monkeypatch.setattr(
        entrypoint_module.os,
        "setgid",
        lambda gid: calls.append(("gid", gid)),
    )
    monkeypatch.setattr(
        entrypoint_module.os,
        "setuid",
        lambda uid: calls.append(("uid", uid)),
    )
    entrypoint_module.drop_runtime_privileges()
    assert calls == [
        ("groups", []),
        ("gid", 1000),
        ("uid", 1000),
    ]


def test_pid1_forwards_sigterm_and_writes_failure(tmp_path: Path) -> None:
    fake_package = tmp_path / "fake-package"
    module_dir = fake_package / "distillery" / "training"
    module_dir.mkdir(parents=True)
    (fake_package / "distillery" / "__init__.py").write_text("", encoding="utf-8")
    (module_dir / "__init__.py").write_text("", encoding="utf-8")
    (module_dir / "entrypoint.py").write_text(
        "\n".join(
            [
                "import os",
                "import signal",
                "import time",
                "from pathlib import Path",
                "",
                "ready = Path(os.environ['FAKE_TRAINER_READY'])",
                "signalled = Path(os.environ['FAKE_TRAINER_SIGNALLED'])",
                "ready.write_text('ready\\n', encoding='utf-8')",
                "",
                "def terminate(_signum, _frame):",
                "    signalled.write_text('sigterm\\n', encoding='utf-8')",
                "    raise SystemExit(143)",
                "",
                "signal.signal(signal.SIGTERM, terminate)",
                "while True:",
                "    time.sleep(0.1)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"training":{"qlora":{"capability_evidence":{}}}}\n',
        encoding="utf-8",
    )
    responses = tmp_path / "responses.jsonl"
    responses.write_text("{}\n", encoding="utf-8")
    models = tmp_path / "models"
    models.mkdir()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "model"
    failure = tmp_path / "output" / "failure"
    ready = tmp_path / "ready"
    signalled = tmp_path / "signalled"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_package)
    env["DISTILLERY_SAGEMAKER_MODEL_INPUT"] = str(models)
    env["DISTILLERY_SAGEMAKER_DATASET_INPUT"] = str(dataset)
    env["DISTILLERY_FAILURE_PATH"] = str(failure)
    env["FAKE_TRAINER_READY"] = str(ready)
    env["FAKE_TRAINER_SIGNALLED"] = str(signalled)
    process = subprocess.Popen(
        [
            sys.executable,
            str(ENTRYPOINT),
            "--manifest",
            str(manifest),
            "--responses",
            str(responses),
            "--output-dir",
            str(output),
            "--model-output-dir",
            str(output),
            "--validate-only",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ready.exists(), process.communicate(timeout=1)
    process.terminate()
    process.communicate(timeout=5)
    assert process.returncode == 143
    assert signalled.read_text(encoding="utf-8") == "sigterm\n"
    failure_text = failure.read_text(encoding="utf-8")
    assert "status 143" in failure_text
    assert "forwarded signal 15" in failure_text
