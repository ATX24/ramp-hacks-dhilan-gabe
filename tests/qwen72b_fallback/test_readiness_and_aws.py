from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.qwen72b_fallback.aws_verifier import (
    MODEL_PREFIX,
    AwsLiveVerifier,
)
from experiments.qwen72b_fallback.bindings import (
    EcrImageBinding,
    load_execution_bindings,
)
from experiments.qwen72b_fallback.memory import require_measured_probe
from experiments.qwen72b_fallback.profile import rehearsal_profile
from experiments.qwen72b_fallback.readiness import (
    ExecutionAction,
    GateCode,
    ReadinessState,
    VerificationFailure,
    evaluate_readiness,
)

ROOT = Path(__file__).resolve().parents[2]
UNUSED_CLIENT = object()


class StreamingBody:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def iter_chunks(self, chunk_size: int) -> Any:
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset : offset + chunk_size]

    def read(self) -> bytes:
        return self.body


class WrongBodyS3:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def list_objects_v2(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "Contents": [
                {"Key": f"{MODEL_PREFIX}/config.json"},
                {"Key": f"{MODEL_PREFIX}/SHA256SUMS"},
                {"Key": f"{MODEL_PREFIX}/snapshot-manifest.json"},
            ]
        }

    def get_object(self, **_kwargs: Any) -> dict[str, Any]:
        return {"Body": StreamingBody(self.body)}


def _verifier(
    *,
    s3: Any = UNUSED_CLIENT,
    ecr: Any = UNUSED_CLIENT,
    open_url: Any = UNUSED_CLIENT,
) -> AwsLiveVerifier:
    return AwsLiveVerifier(
        repo_root=ROOT,
        sts=object(),
        s3=s3,
        ecr=ecr,
        iam=object(),
        ec2=object(),
        sagemaker=object(),
        **({} if open_url is UNUSED_CLIENT else {"open_url": open_url}),
    )


def test_live_result_has_authorization_or_blockers_never_forgeable_may_execute(
    live_verifier_factory,
) -> None:
    profile = rehearsal_profile()
    verifier = live_verifier_factory(
        action=ExecutionAction.REHEARSAL,
        profile=profile,
    )

    def blocked_reviews() -> Any:
        raise VerificationFailure(
            GateCode.EXECUTION_REVIEWS,
            "both review packets pending",
        )

    verifier.verify_reviews = blocked_reviews  # type: ignore[method-assign]
    report = evaluate_readiness(
        verifier,
        action=ExecutionAction.REHEARSAL,
        launch_name="qwen72b-rehearsal-test",
        profile=profile,
        typed_confirmation=("EXECUTE QWEN72B REHEARSAL qwen72b-rehearsal-test"),
    )
    payload = report.model_dump(mode="json")
    assert report.state is ReadinessState.BLOCKED
    assert report.authorization is None
    assert "may_execute" not in payload
    assert GateCode.EXECUTION_REVIEWS in {blocked.gate for blocked in report.blocked_gates}


def test_committed_execution_bindings_keep_both_reviews_and_resources_blocked() -> None:
    bindings = load_execution_bindings()
    assert bindings.review_packet_sha256 == ()
    assert bindings.ecr_image is None
    assert bindings.memory_probe is None
    assert bindings.transfer_ami_id is None
    with pytest.raises(VerificationFailure, match="review packet"):
        _verifier().verify_reviews()


def test_same_size_wrong_s3_bytes_fail_body_hash_gate(monkeypatch) -> None:
    expected_body = b"good"
    wrong_body = b"evil"
    assert len(expected_body) == len(wrong_body)
    inventory = SimpleNamespace(
        files={
            "config.json": SimpleNamespace(
                size=len(expected_body),
                sha256=hashlib.sha256(expected_body).hexdigest(),
            )
        },
        inventory_sha256="a" * 64,
    )
    monkeypatch.setattr(
        "experiments.qwen72b_fallback.aws_verifier.load_weight_inventory",
        lambda: inventory,
    )
    with pytest.raises(VerificationFailure, match="body mismatch"):
        _verifier(s3=WrongBodyS3(wrong_body)).verify_s3_snapshot()


def test_wrong_ecr_digest_fails_exact_image_gate(monkeypatch) -> None:
    binding = EcrImageBinding(
        image_digest="sha256:" + ("1" * 64),
        image_uri=(
            "225989358036.dkr.ecr.us-east-1.amazonaws.com/distillery-training@sha256:" + ("1" * 64)
        ),
        source_revision="a" * 40,
        package_lock_sha256="b" * 64,
        source_tree_sha256="c" * 64,
        qwen72b_trainer_packaged=True,
    )
    monkeypatch.setattr(
        "experiments.qwen72b_fallback.aws_verifier.load_execution_bindings",
        lambda: SimpleNamespace(ecr_image=binding),
    )

    class Ecr:
        def describe_repositories(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "repositories": [
                    {
                        "repositoryUri": (
                            "225989358036.dkr.ecr.us-east-1.amazonaws.com/distillery-training"
                        )
                    }
                ]
            }

        def describe_images(self, **_kwargs: Any) -> dict[str, Any]:
            return {"imageDetails": [{"imageDigest": "sha256:" + ("2" * 64)}]}

    with pytest.raises(VerificationFailure, match="digest"):
        _verifier(ecr=Ecr()).verify_ecr_image()


def test_exact_digest_without_sealed_trainer_labels_fails_image_gate(
    monkeypatch,
) -> None:
    digest = "sha256:" + ("1" * 64)
    binding = EcrImageBinding(
        image_digest=digest,
        image_uri=(f"225989358036.dkr.ecr.us-east-1.amazonaws.com/distillery-training@{digest}"),
        source_revision="a" * 40,
        package_lock_sha256="b" * 64,
        source_tree_sha256="c" * 64,
        qwen72b_trainer_packaged=True,
    )
    monkeypatch.setattr(
        "experiments.qwen72b_fallback.aws_verifier.load_execution_bindings",
        lambda: SimpleNamespace(ecr_image=binding),
    )
    config_body = json.dumps(
        {"config": {"Labels": {"distillery.qwen72b.trainer": "wrong.module"}}},
        sort_keys=True,
    ).encode()
    config_digest = "sha256:" + hashlib.sha256(config_body).hexdigest()

    class Ecr:
        def describe_repositories(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "repositories": [
                    {
                        "repositoryUri": (
                            "225989358036.dkr.ecr.us-east-1.amazonaws.com/distillery-training"
                        )
                    }
                ]
            }

        def describe_images(self, **_kwargs: Any) -> dict[str, Any]:
            return {"imageDetails": [{"imageDigest": digest}]}

        def batch_get_image(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "images": [
                    {
                        "imageId": {"imageDigest": digest},
                        "imageManifest": json.dumps({"config": {"digest": config_digest}}),
                    }
                ]
            }

        def get_download_url_for_layer(self, **_kwargs: Any) -> dict[str, str]:
            return {"downloadUrl": "https://example.invalid/config"}

    with pytest.raises(VerificationFailure, match="labels"):
        _verifier(
            ecr=Ecr(),
            open_url=lambda *_args, **_kwargs: io.BytesIO(config_body),
        ).verify_ecr_image()


def test_formula_never_authorizes_ddp_without_measured_probe() -> None:
    with pytest.raises(RuntimeError, match="measured 72B QLoRA"):
        require_measured_probe(
            None,
            profile_sha256="1" * 64,
            model_identity_sha256="2" * 64,
            image_binding_sha256="3" * 64,
            runtime_image_digest="sha256:" + ("4" * 64),
        )


def test_gate_command_maps_blocked_state_to_nonzero_exit() -> None:
    source = (ROOT / "scripts" / "qwen72b" / "check_gates.py").read_text(encoding="utf-8")
    assert "if report.state is ReadinessState.BLOCKED:" in source
    assert "return 3" in source
