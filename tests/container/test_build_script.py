"""Executable packaging, source-binding, and ML compatibility regressions."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from conftest import (
    BUILD_SCRIPT,
    ML_COMPATIBILITY,
    REPO,
    STAGE_TOOL,
    VERIFY_ML,
    load_module,
)


def run_stage(destination: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(BUILD_SCRIPT), f"--stage-only={destination}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def uv_sync_smoke(stage: Path, environment: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(environment)
    lock = tomllib.loads((stage / "uv.lock").read_text(encoding="utf-8"))
    external_packages = sorted(
        {package["name"] for package in lock["package"] if package["name"] != "distillery"}
    )
    excluded_install_args = [
        argument for package in external_packages for argument in ("--no-install-package", package)
    ]
    return subprocess.run(
        [
            "uv",
            "sync",
            "--frozen",
            "--no-dev",
            "--extra",
            "ml",
            "--no-editable",
            "--offline",
            "--no-python-downloads",
            *excluded_install_args,
        ],
        cwd=stage,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_packaging_scripts_are_executable() -> None:
    for path in (
        BUILD_SCRIPT,
        REPO / "scripts" / "container" / "manifest_tool.py",
        REPO / "scripts" / "container" / "stage_context.py",
        VERIFY_ML,
    ):
        assert path.stat().st_mode & stat.S_IXUSR


def test_staging_rejects_credentials_weights_and_symlinks(tmp_path: Path) -> None:
    stage_module = load_module(STAGE_TOOL, "distillery_stage_context_security")
    for name in ("credentials", ".env", "private.pem", "weights.safetensors"):
        path = tmp_path / name
        path.write_text("not-real-sensitive-data\n", encoding="utf-8")
        with pytest.raises(ValueError):
            stage_module.ensure_safe_source(path)

    target = tmp_path / "target.py"
    target.write_text("pass\n", encoding="utf-8")
    symlink = tmp_path / "linked.py"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        stage_module.ensure_safe_source(symlink)


def test_stage_context_is_reproducible_and_complete(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = run_stage(first)
    second_result = run_stage(second)
    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr

    first_inventory = json.loads((first / "SOURCE_FILES.json").read_text(encoding="utf-8"))
    second_inventory = json.loads((second / "SOURCE_FILES.json").read_text(encoding="utf-8"))
    assert first_inventory == second_inventory
    staged_paths = {record["path"] for record in first_inventory["files"]}
    assert {
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "uv.lock",
        ".dockerignore",
        "containers/training/Dockerfile",
        "containers/training/container_entrypoint.py",
        "containers/training/ml-compatibility.json",
        "containers/training/verify_ml_compatibility.py",
    }.issubset(staged_paths)
    assert any(path.startswith("src/distillery/") for path in staged_paths)
    assert not any("__pycache__" in path for path in staged_paths)
    assert not any(path.startswith(("apps/", "tests/", "docs/", ".git/")) for path in staged_paths)
    for path in first.rglob("*"):
        assert path.stat().st_mtime == 0
        expected_mode = 0o755 if path.is_dir() else 0o644
        assert stat.S_IMODE(path.stat().st_mode) == expected_mode


def test_staged_uv_sync_smoke_validates_package_metadata(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    result = run_stage(stage)
    assert result.returncode == 0, result.stderr

    smoke = uv_sync_smoke(stage, tmp_path / "environment")
    assert smoke.returncode == 0, smoke.stderr

    incomplete_stage = tmp_path / "incomplete-stage"
    incomplete_result = run_stage(incomplete_stage)
    assert incomplete_result.returncode == 0, incomplete_result.stderr
    pyproject = incomplete_stage / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            'readme = "README.md"',
            'readme = "PACKAGING-README-MISSING.md"',
        ),
        encoding="utf-8",
    )
    missing_metadata = uv_sync_smoke(
        incomplete_stage,
        tmp_path / "missing-environment",
    )
    assert missing_metadata.returncode != 0
    error = (missing_metadata.stdout + missing_metadata.stderr).lower()
    assert "packaging-readme-missing.md" in error or "readme" in error


def test_current_foundation_lock_matches_declared_stack() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_ML),
            "lock",
            "--lock",
            str(REPO / "uv.lock"),
            "--compatibility",
            str(ML_COMPATIBILITY),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "compatible"
    assert payload["actual"]["torch_version"] == "2.4.1"
    assert payload["actual"]["bitsandbytes_version"] == "0.44.1"
    assert payload["actual"]["forbidden_packages"] == []
    assert payload["actual"]["unexpected_accelerator_packages"] == []


def test_compatible_lock_contract_passes(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text(
        "\n".join(
            [
                "version = 1",
                "",
                "[[package]]",
                'name = "torch"',
                'version = "2.4.1"',
                "wheels = [",
                (
                    '  { url = "https://download.pytorch.org/whl/cu124/'
                    'torch-2.4.1-cp311-cp311-manylinux_2_28_x86_64.whl" },'
                ),
                "]",
                "",
                "[[package]]",
                'name = "bitsandbytes"',
                'version = "0.44.1"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_ML),
            "lock",
            "--lock",
            str(lock),
            "--compatibility",
            str(ML_COMPATIBILITY),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "compatible"


def test_default_dry_run_records_plan_without_fake_registry_digest(tmp_path: Path) -> None:
    manifest = tmp_path / "build manifest.json"
    env = os.environ.copy()
    env["DISTILLERY_BUILD_MANIFEST"] = str(manifest)
    env.pop("AWS_PROFILE", None)
    result = subprocess.run(
        [str(BUILD_SCRIPT), "--dry-run"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "docker build not invoked" in result.stdout
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["ml_compatibility"]["status"] == "compatible"
    assert payload["local"]["config_id"] is None
    assert payload["registry"]["image_digest"] is None
    assert payload["registry"]["digest_uri"] is None
    assert payload["registry"]["verified"] is False


def test_reviewed_sha_cannot_relabel_dirty_worktree(tmp_path: Path) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    manifest = tmp_path / "manifest.json"
    env = os.environ.copy()
    env["DISTILLERY_BUILD_MANIFEST"] = str(manifest)
    env.pop("AWS_PROFILE", None)
    result = subprocess.run(
        [str(BUILD_SCRIPT), "--dry-run", f"--reviewed-source-sha={head}"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["source"]["reviewed_commit_sha"] == head
    assert payload["source"]["clean"] is False
    assert payload["source"]["commit_bound"] is False


def test_real_build_refuses_dirty_source_before_docker(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "docker-called"
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        f"#!/usr/bin/env bash\ntouch {marker!s}\nexit 99\n",
        encoding="utf-8",
    )
    fake_docker.chmod(fake_docker.stat().st_mode | stat.S_IXUSR)
    head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env.pop("AWS_PROFILE", None)
    result = subprocess.run(
        [str(BUILD_SCRIPT), "--build", f"--reviewed-source-sha={head}"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "clean committed source tree" in result.stderr
    assert not marker.exists()


def test_real_build_stages_reviewed_commit_not_ignored_worktree_content(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    for directory in (
        repo / "containers" / "training",
        repo / "scripts" / "container",
        repo / "src" / "distillery",
    ):
        directory.mkdir(parents=True)
    for name in ("README.md", "LICENSE", "pyproject.toml", "uv.lock"):
        shutil.copy2(REPO / name, repo / name)
    shutil.copy2(
        REPO / "src" / "distillery" / "__init__.py",
        repo / "src" / "distillery" / "__init__.py",
    )
    for source in (REPO / "containers" / "training").iterdir():
        if source.is_file():
            shutil.copy2(source, repo / "containers" / "training" / source.name)
    for name in ("build_training_image.sh", "manifest_tool.py", "stage_context.py"):
        shutil.copy2(
            REPO / "scripts" / "container" / name,
            repo / "scripts" / "container" / name,
        )
    (repo / ".gitignore").write_text(
        "src/distillery/ignored.py\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Container Tests",
            "-c",
            "user.email=container-tests@example.invalid",
            "commit",
            "-qm",
            "reviewed source",
        ],
        check=True,
    )
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    (repo / "src" / "distillery" / "ignored.py").write_text(
        "arbitrary_dirty_content = True\n",
        encoding="utf-8",
    )
    assert (
        subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"],
            text=True,
        )
        == ""
    )

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    marker = tmp_path / "docker-context-result"
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

args = sys.argv[1:]
if args[0] == "build":
    context = Path(args[-1])
    result = "present" if (context / "src/distillery/ignored.py").exists() else "absent"
    Path(os.environ["FAKE_CONTEXT_RESULT"]).write_text(result + "\\n", encoding="utf-8")
elif args[:2] == ["image", "inspect"]:
    print("sha256:" + ("1" * 64))
else:
    raise SystemExit(91)
""",
        encoding="utf-8",
    )
    fake_docker.chmod(fake_docker.stat().st_mode | stat.S_IXUSR)
    manifest = tmp_path / "real-build-manifest.json"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "DISTILLERY_BUILD_MANIFEST": str(manifest),
            "DISTILLERY_PACKAGING_PYTHON": sys.executable,
            "FAKE_CONTEXT_RESULT": str(marker),
        }
    )
    result = subprocess.run(
        [
            str(repo / "scripts" / "container" / "build_training_image.sh"),
            "--build",
            f"--reviewed-source-sha={head}",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert marker.read_text(encoding="utf-8") == "absent\n"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["source"]["commit_sha"] == head
    assert payload["source"]["reviewed_commit_sha"] == head
    assert payload["source"]["commit_bound"] is True
    assert payload["local"]["config_id"] == "sha256:" + ("1" * 64)
    assert payload["registry"]["image_digest"] is None


def test_nonexistent_reviewed_commit_is_rejected() -> None:
    result = subprocess.run(
        [str(BUILD_SCRIPT), "--build", f"--reviewed-source-sha={'0' * 40}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "not an existing git commit" in result.stderr


def test_selected_root_profile_is_rejected(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_aws = fake_bin / "aws"
    fake_aws.write_text(
        "#!/usr/bin/env bash\n"
        'echo \'{"Account":"123456789012",'
        '"Arn":"arn:aws:iam::123456789012:root"}\'\n',
        encoding="utf-8",
    )
    fake_aws.chmod(fake_aws.stat().st_mode | stat.S_IXUSR)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["AWS_PROFILE"] = "unsafe"
    result = subprocess.run(
        [str(BUILD_SCRIPT), "--dry-run"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "account root identity" in result.stderr
