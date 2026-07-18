#!/usr/bin/env python3
"""Bounded emergency rescue: one real oracle_sft smoke job via Script Mode DLC.

Uses the pinned AWS Hugging Face training DLC digest plus a sealed source bundle
from the current committed tree. Refuses CreateTrainingJob when another Distillery
job is Starting/InProgress. Mutation requires --execute, DISTILLERY_AWS_SMOKE_EXECUTE=1,
and the exact confirmation phrase.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from distillery.contracts.hashing import content_sha256, sha256_hex  # noqa: E402
from experiments.aws_smoke.channels import load_manifest  # noqa: E402
from experiments.aws_smoke.dataset_subset import materialize_emergency_subset  # noqa: E402
from experiments.aws_smoke.launch_plan import stage_manifest_for_job  # noqa: E402
from experiments.aws_smoke.manifests import write_arm_manifests  # noqa: E402
from experiments.aws_smoke.pins import EmergencyEvidence, load_evidence  # noqa: E402
from experiments.aws_smoke.profile import DEFAULT_EMERGENCY_PROFILE  # noqa: E402
from experiments.aws_smoke.rescue_launch import (  # noqa: E402
    RESCUE_DLC_DIGEST,
    RESCUE_DLC_ECR_ARN,
    RESCUE_DLC_IMAGE_URI,
    RESCUE_DLC_TAG,
    build_rescue_create_training_job_request,
    list_active_distillery_jobs,
)
from experiments.aws_smoke.safety import CONFIRM_PHRASE, CallerIdentity  # noqa: E402
from experiments.aws_smoke.tokenization import load_tokenization_evidence  # noqa: E402
from infra.sagemaker.role import training_role_inline_policy  # noqa: E402

BUCKET = "distillery-225989358036-us-east-1"
ACCOUNT = "225989358036"
ROLE_NAME = "distillery-sagemaker-training"
STUDENT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TEACHER_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
STUDENT_CONFIG_SHA = "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"
TEACHER_CONFIG_SHA = "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670"
MATERIALIZATION_SHA = "7e6d59fa8d30805e168a52bc4ef4225e2dff900159489c71cf62e5dde9fd70e6"
SOURCE_PATHS = (
    "experiments/__init__.py",
    "experiments/aws_smoke",
    "src/distillery",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
)


def _sts_identity(profile: str, region: str) -> CallerIdentity:
    raw = boto3.Session(profile_name=profile, region_name=region).client(
        "sts"
    ).get_caller_identity()
    return CallerIdentity(
        account=str(raw["Account"]),
        arn=str(raw["Arn"]),
        user_id=str(raw["UserId"]),
    )


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _require_clean_tree() -> None:
    status = subprocess.check_output(
        ["git", "-C", str(_REPO_ROOT), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if status:
        raise SystemExit(
            "rescue source bundle requires a clean committed tree; dirty paths:\n"
            + status
        )


def _build_source_bundle(work_dir: Path, *, source_revision: str) -> Path:
    bundle_root = work_dir / "code"
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True)
    archive = work_dir / "source.git.tar"
    subprocess.check_call(
        [
            "git",
            "-C",
            str(_REPO_ROOT),
            "archive",
            "--format=tar",
            source_revision,
            *SOURCE_PATHS,
        ],
        stdout=archive.open("wb"),
    )
    with tarfile.open(archive, "r:") as handle:
        handle.extractall(bundle_root)
    # rescue_entry must be present from the archive after commit; copy from tree
    # only when it exists at HEAD (committed).
    entry = bundle_root / "experiments" / "aws_smoke" / "rescue_entry.py"
    if not entry.is_file():
        raise FileNotFoundError(
            "experiments/aws_smoke/rescue_entry.py missing from committed archive"
        )
    # Flat entrypoint expected by ContainerEntrypoint.
    shutil.copy2(entry, bundle_root / "rescue_entry.py")
    wheels_src = Path("/tmp/distillery-rescue/wheels")
    if not wheels_src.is_dir() or not list(wheels_src.glob("*.whl")):
        raise FileNotFoundError(
            "offline wheels missing under /tmp/distillery-rescue/wheels; "
            "download pydantic/rfc8785 wheels before launch"
        )
    wheels_dst = bundle_root / "wheels"
    shutil.copytree(wheels_src, wheels_dst)
    # Local staged import/argument smoke against the sealed bundle.
    subprocess.check_call(
        [
            sys.executable,
            str(bundle_root / "rescue_entry.py"),
            "--help",
        ],
        cwd=str(bundle_root),
        env={
            **os.environ,
            "PYTHONPATH": f"{bundle_root}:{bundle_root / 'src'}",
        },
    )
    marker = bundle_root / "SOURCE_REVISION.txt"
    marker.write_text(source_revision + "\n", encoding="utf-8")
    return bundle_root


def _upload_prefix(s3: Any, local_dir: Path, bucket: str, prefix: str) -> str:
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        key = f"{prefix.rstrip('/')}/{path.relative_to(local_dir).as_posix()}"
        extra = {"ContentType": "application/octet-stream"}
        if path.suffix == ".json":
            extra["ContentType"] = "application/json"
        elif path.suffix == ".jsonl":
            extra["ContentType"] = "application/jsonlines"
        elif path.suffix == ".py":
            extra["ContentType"] = "text/x-python"
        s3.upload_file(str(path), bucket, key, ExtraArgs=extra)
    return f"s3://{bucket}/{prefix.rstrip('/')}/"


def _apply_scoped_role_policy(
    *,
    iam: Any,
    run_artifact_prefix: str,
    dataset_prefix: str,
    code_prefix: str,
) -> dict[str, Any]:
    policy = training_role_inline_policy(
        artifact_bucket_arn=f"arn:aws:s3:::{BUCKET}",
        model_bucket_arn=f"arn:aws:s3:::{BUCKET}",
        ecr_repository_arn=(
            f"arn:aws:ecr:us-east-1:{ACCOUNT}:repository/distillery-training"
        ),
        run_artifact_prefix=run_artifact_prefix,
        dataset_prefix=dataset_prefix,
        model_channel_prefix="models",
        model_prefix="models/Qwen",
        model_materialization_key="models/materialization.json",
        code_prefix=code_prefix,
        additional_ecr_repository_arns=(RESCUE_DLC_ECR_ARN,),
    )
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="DistillerySmokeTrainingLeastPrivilege",
        PolicyDocument=json.dumps(policy),
    )
    return policy


def _wait_in_progress(sm: Any, logs: Any, job_name: str, *, timeout_s: int = 900) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_status = ""
    log_evidence: list[str] = []
    while time.time() < deadline:
        desc = sm.describe_training_job(TrainingJobName=job_name)
        status = str(desc.get("TrainingJobStatus", "Unknown"))
        secondary = str(desc.get("SecondaryStatus", ""))
        if status != last_status:
            print(
                json.dumps(
                    {
                        "event": "job_status",
                        "job_name": job_name,
                        "status": status,
                        "secondary_status": secondary,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_status = status
        if status == "InProgress":
            log_evidence = _collect_log_evidence(logs, job_name)
            return {"description": desc, "log_evidence": log_evidence}
        if status in {"Failed", "Stopped"}:
            log_evidence = _collect_log_evidence(logs, job_name)
            raise SystemExit(
                json.dumps(
                    {
                        "ok": False,
                        "status": status,
                        "failure_reason": desc.get("FailureReason"),
                        "log_evidence": log_evidence,
                    },
                    sort_keys=True,
                    indent=2,
                )
            )
        time.sleep(15)
    raise SystemExit(f"timed out waiting for InProgress on {job_name}")


def _collect_log_evidence(logs: Any, job_name: str) -> list[str]:
    group = "/aws/sagemaker/TrainingJobs"
    evidence: list[str] = []
    try:
        streams = logs.describe_log_streams(
            logGroupName=group,
            logStreamNamePrefix=job_name,
            orderBy="LogStreamName",
            descending=True,
            limit=5,
        )
    except Exception as exc:  # noqa: BLE001
        return [f"log_stream_lookup_failed:{exc}"]
    for stream in streams.get("logStreams", []):
        name = stream.get("logStreamName")
        if not name:
            continue
        events = logs.get_log_events(
            logGroupName=group,
            logStreamName=name,
            startFromHead=True,
            limit=100,
        )
        for event in events.get("events", []):
            message = str(event.get("message", "")).strip()
            if not message:
                continue
            lowered = message.lower()
            if any(
                token in lowered
                for token in (
                    "emergency_trainer_start",
                    "rescue_import_smoke_ok",
                    "optimizer",
                    "adamw",
                    "loading checkpoint",
                    "loaded",
                    "step ",
                    "train_step",
                    "qlora",
                    "peft",
                    "model",
                )
            ):
                evidence.append(message[:500])
    return evidence[:40]


def main() -> int:
    parser = argparse.ArgumentParser(prog="launch_oracle_sft_rescue")
    parser.add_argument("--profile", default="gabriel-cli")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/distillery-rescue/run"))
    parser.add_argument("--models-dir", type=Path, default=Path("/tmp/distillery-rescue/models"))
    parser.add_argument("--tokenization-evidence", type=Path, default=None)
    parser.add_argument("--confirm", type=str, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-role-patch", action="store_true")
    args = parser.parse_args()

    if args.profile != "gabriel-cli":
        raise SystemExit("profile must be gabriel-cli")
    source_revision = _git_head()
    _require_clean_tree()

    tok_path = args.tokenization_evidence or Path(
        "/tmp/distillery-rescue/tokenizer_evidence.json"
    )
    if not tok_path.is_file():
        raise SystemExit(f"missing tokenizer evidence: {tok_path}")
    tok_raw = json.loads(tok_path.read_text(encoding="utf-8"))
    student = tok_raw["student"]
    teacher = tok_raw["teacher"]

    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    subset = materialize_emergency_subset(work_dir / "subset")
    evidence_payload = {
        "schema_version": "distillery.aws_smoke.evidence.v1",
        "aws_account_id": ACCOUNT,
        "aws_region": args.region,
        "aws_profile": "gabriel-cli",
        "iam_role_arn": f"arn:aws:iam::{ACCOUNT}:role/{ROLE_NAME}",
        "artifact_s3_prefix": f"s3://{BUCKET}/artifacts/rescue",
        "dataset_s3_uri": f"s3://{BUCKET}/datasets/ds_awssmoke01",
        "models_s3_uri": f"s3://{BUCKET}/models",
        "model_materialization_uri": f"s3://{BUCKET}/models/materialization.json",
        "model_materialization_sha256": MATERIALIZATION_SHA,
        "ecr_image_uri": RESCUE_DLC_IMAGE_URI,
        "image_digest": RESCUE_DLC_DIGEST,
        "student_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "student_revision": STUDENT_REVISION,
        "student_model_config_sha256": STUDENT_CONFIG_SHA,
        "teacher_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "teacher_revision": TEACHER_REVISION,
        "teacher_model_config_sha256": TEACHER_CONFIG_SHA,
        "student_tokenizer_sha256": student["tokenizer_sha256"],
        "teacher_tokenizer_sha256": teacher["tokenizer_sha256"],
        "student_chat_template_sha256": student["chat_template_sha256"],
        "teacher_chat_template_sha256": teacher["chat_template_sha256"],
        "student_special_token_map": student["special_token_map"],
        "teacher_special_token_map": teacher["special_token_map"],
        "package_lock_hash": sha256_hex((_REPO_ROOT / "uv.lock").read_bytes()),
        "source_revision": source_revision,
        "proof_protocol_id": "finance-proof.v1",
        "proof_protocol_sha256": content_sha256({"id": "finance-proof.v1"}),
        "license_disposition": (
            "Apache-2.0; output-use accepted for hackathon smoke with counsel follow-up"
        ),
        "output_use_disposition": "synthetic_demo_outputs_ok_no_customer_data",
        "data_content_sha256": subset.content_sha256,
        "price_source": "operator_attested_ml.g5.xlarge_us-east-1_1.408",
        "hourly_usd": 1.408,
        "evidence_attested_by": "gabriel-rescue",
        "evidence_notes": (
            f"Script Mode rescue via pinned HF DLC {RESCUE_DLC_TAG}; "
            "oracle_sft only; no teacher generation."
        ),
        "memory_probe_evidence": None,
    }
    evidence_path = work_dir / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evidence = load_evidence(evidence_path)

    # Build tokenization evidence for oracle_sft via existing script path.
    tokenization_out = work_dir / "tokenization.json"
    subprocess.check_call(
        [
            str(_REPO_ROOT / ".venv-synth" / "bin" / "python"),
            str(_REPO_ROOT / "scripts" / "aws_smoke" / "materialize_tokenization_evidence.py"),
            "--evidence",
            str(evidence_path),
            "--subset-dir",
            str(work_dir / "subset"),
            "--models-dir",
            str(args.models_dir),
            "--output",
            str(tokenization_out),
        ],
        cwd=str(_REPO_ROOT),
        env={
            **os.environ,
            "PYTHONPATH": f"{_REPO_ROOT}:{_REPO_ROOT / 'src'}",
        },
    )
    tokenization = load_tokenization_evidence(tokenization_out)
    train_rows = [
        json.loads(line)
        for line in subset.train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    paths = write_arm_manifests(
        output_dir=work_dir / "manifests",
        evidence=evidence,
        dataset_id="ds_awssmoke01",
        dataset_uri=evidence.dataset_s3_uri,
        dataset_sha256=subset.content_sha256,
        split_sha256=subset.split_sha256,
        example_ids=[str(row["example_id"]) for row in train_rows],
        tasks=[str(row["task"]) for row in train_rows],
        difficulties=[str(row["difficulty"]) for row in train_rows],
        tokenization_evidence=tokenization,
        arms=("oracle_sft",),
        profile=DEFAULT_EMERGENCY_PROFILE,
    )
    manifest = load_manifest(paths["oracle_sft"])
    manifest_sha = manifest.seal_sha256()
    max_cost = float(manifest.cost.max_run_usd)
    if max_cost > 50.0:
        raise SystemExit(f"max cost {max_cost} exceeds $50 rescue ceiling")

    # Nonempty responses channel stub (oracle_sft does not consume teacher targets).
    responses_dir = work_dir / "dataset_upload" / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    (responses_dir / "responses.jsonl").write_text(
        json.dumps(
            {
                "schema": "distillery.responses.channel.v1",
                "arm": "oracle_sft",
                "note": "unused_by_oracle_sft_hard_targets",
                "manifest_sha256": manifest_sha,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_upload = work_dir / "dataset_upload"
    shutil.copy2(subset.train_path, dataset_upload / "train.jsonl")
    shutil.copy2(subset.validation_path, dataset_upload / "validation.jsonl")
    shutil.copy2(subset.subset_manifest_path, dataset_upload / "subset_manifest.json")

    code_root = _build_source_bundle(work_dir, source_revision=source_revision)
    code_prefix = f"artifacts/rescue/code/{source_revision[:12]}-{manifest_sha[:12]}"
    run_prefix = f"artifacts/rescue/runs/{manifest.run_id}"
    dataset_prefix = "datasets/ds_awssmoke01"

    request = build_rescue_create_training_job_request(
        manifest=manifest,
        evidence=evidence,
        code_s3_uri=f"s3://{BUCKET}/{code_prefix}",
    )
    plan = {
        "dry_run": not args.execute,
        "job_name": request["TrainingJobName"],
        "manifest_sha256": manifest_sha,
        "image_digest": RESCUE_DLC_DIGEST,
        "image_uri": RESCUE_DLC_IMAGE_URI,
        "dlc_tag": RESCUE_DLC_TAG,
        "model_revision": STUDENT_REVISION,
        "output_uri": request["OutputDataConfig"]["S3OutputPath"],
        "max_run_usd": max_cost,
        "max_runtime_seconds": DEFAULT_EMERGENCY_PROFILE.max_runtime_seconds,
        "source_revision": source_revision,
        "code_s3_uri": f"s3://{BUCKET}/{code_prefix}/",
        "create_training_job_request": request,
    }
    plan_path = work_dir / "rescue_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "mode": "planned", "plan": plan}, sort_keys=True))

    if not args.execute:
        return 0
    if os.environ.get("DISTILLERY_AWS_SMOKE_EXECUTE", "") != "1":
        raise SystemExit("set DISTILLERY_AWS_SMOKE_EXECUTE=1 to submit")
    if args.confirm != CONFIRM_PHRASE:
        raise SystemExit(f"missing confirmation phrase {CONFIRM_PHRASE}")

    identity = _sts_identity(args.profile, args.region)
    if identity.account != ACCOUNT or identity.arn.endswith(":root"):
        raise SystemExit(f"refusing submit with identity {identity.arn}")

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    sm = session.client("sagemaker")
    s3 = session.client("s3")
    iam = session.client("iam")
    logs = session.client("logs")

    active = list_active_distillery_jobs(sm)
    if active:
        print(
            json.dumps(
                {
                    "ok": False,
                    "mode": "duplicate_prevented",
                    "active_jobs": active,
                    "note": "another Distillery job is Starting/InProgress; not submitting",
                },
                sort_keys=True,
                indent=2,
            )
        )
        # Verify and report the first active job instead of spending.
        first = active[0]["name"]
        desc = sm.describe_training_job(TrainingJobName=first)
        evidence_logs = _collect_log_evidence(logs, first)
        print(
            json.dumps(
                {
                    "existing_job_arn": desc.get("TrainingJobArn"),
                    "existing_job_name": first,
                    "existing_job_status": desc.get("TrainingJobStatus"),
                    "existing_image": desc.get("AlgorithmSpecification", {}).get(
                        "TrainingImage"
                    ),
                    "existing_output": desc.get("OutputDataConfig", {}).get("S3OutputPath"),
                    "log_evidence": evidence_logs,
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    if not args.skip_role_patch:
        _apply_scoped_role_policy(
            iam=iam,
            run_artifact_prefix=run_prefix,
            dataset_prefix=dataset_prefix,
            code_prefix=code_prefix,
        )

    _upload_prefix(s3, dataset_upload, BUCKET, dataset_prefix)
    _upload_prefix(s3, code_root, BUCKET, code_prefix)
    staged = stage_manifest_for_job(
        s3,
        local_manifest_path=paths["oracle_sft"],
        request=request,
    )

    try:
        sm.create_training_job(**request)
    except Exception as exc:  # noqa: BLE001
        # Deterministic name: concurrent create loses safely.
        if "ResourceInUse" in str(exc) or "Cannot create already existing" in str(exc):
            desc = sm.describe_training_job(TrainingJobName=request["TrainingJobName"])
            print(
                json.dumps(
                    {
                        "ok": True,
                        "mode": "lost_create_race",
                        "job_arn": desc.get("TrainingJobArn"),
                        "job_name": request["TrainingJobName"],
                        "status": desc.get("TrainingJobStatus"),
                    },
                    sort_keys=True,
                )
            )
            return 0
        raise

    waited = _wait_in_progress(sm, logs, request["TrainingJobName"])
    desc = waited["description"]
    result = {
        "ok": True,
        "mode": "submitted",
        "job_arn": desc.get("TrainingJobArn"),
        "job_name": request["TrainingJobName"],
        "status": desc.get("TrainingJobStatus"),
        "secondary_status": desc.get("SecondaryStatus"),
        "manifest_sha256": manifest_sha,
        "image_digest": RESCUE_DLC_DIGEST,
        "image_uri": RESCUE_DLC_IMAGE_URI,
        "model_revision": STUDENT_REVISION,
        "output_uri": request["OutputDataConfig"]["S3OutputPath"],
        "max_run_usd": max_cost,
        "manifest_object_uri": staged,
        "code_s3_uri": f"s3://{BUCKET}/{code_prefix}/",
        "source_revision": source_revision,
        "log_evidence": waited["log_evidence"],
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    (work_dir / "submit_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
