"""Readiness gates and plan-only script tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.qwen72b_fallback.readiness import evaluate_readiness

ROOT = Path(__file__).resolve().parents[2]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}"
    return env


def test_readiness_blocks_on_active_14b_and_missing_ecr() -> None:
    report = evaluate_readiness(
        action="rehearsal",
        identity_ok=True,
        inventory_ok=True,
        license_ok=True,
        tokenizer_family_ok=True,
        iam_transfer_role_ok=True,
        iam_training_role_ok=True,
        ecr_image_digest_present=False,
        snapshot_complete_on_s3=False,
        conflicting_p4de_job_active=False,
        conflicting_transfer_active=True,
        active_g5_smoke=False,
        active_14b_work=True,
        materialization_projected_usd=10.0,
    )
    assert report.may_execute is False
    reasons = " ".join(report.blocking_reasons).lower()
    assert "ecr" in reasons
    assert "14b" in reasons or "transfer" in reasons or "g5" in reasons


def test_materialize_readiness_blocks_conflicting_transfer() -> None:
    report = evaluate_readiness(
        action="materialize",
        identity_ok=True,
        inventory_ok=True,
        license_ok=True,
        tokenizer_family_ok=True,
        iam_transfer_role_ok=True,
        iam_training_role_ok=True,
        ecr_image_digest_present=False,
        snapshot_complete_on_s3=False,
        conflicting_p4de_job_active=False,
        conflicting_transfer_active=True,
        active_g5_smoke=False,
        active_14b_work=True,
        materialization_projected_usd=25.0,
    )
    assert report.may_execute is False


def test_materialize_plan_script() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/qwen72b/materialize_to_s3.py"),
            "--mode",
            "plan",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=_env(),
    )
    plan = json.loads(proc.stdout)
    assert plan["hard_cap_usd"] == 500.0
    assert plan["revision"] == "495f39366efef23836d0cfae4fbe635880d2be31"
    assert plan["verify_checksums"] is True


def test_execute_latches_refuse() -> None:
    for script in (
        ROOT / "scripts/qwen72b/materialize_to_s3.py",
        ROOT / "scripts/qwen72b/launch_rehearsal.py",
    ):
        proc = subprocess.run(
            [sys.executable, str(script), "--execute"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=_env(),
        )
        assert proc.returncode == 2
        payload = json.loads(proc.stdout)
        assert payload["ok"] is False


def test_rehearsal_plan_script() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/qwen72b/launch_rehearsal.py"),
            "--mode",
            "plan",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=_env(),
    )
    plan = json.loads(proc.stdout)
    assert plan["optimizer_steps"] == 3
    assert plan["profile"]["precision_mode"] == "qlora_4bit"
    assert (
        plan["roles"]["oracle_sft_adapted_fallback"]["may_be_called_distilled_student"]
        is False
    )
    assert plan["trajectories"]["included_in_warm_timer"] is False
