"""Executable Qwen model-license obligations, separate from MIT repo code."""

from __future__ import annotations

import json
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field

from experiments.qwen72b_fallback.evidence import (
    SHA256_PATTERN,
    HashBoundEvidence,
    VerificationSource,
    sha256_bytes,
    sha256_file,
)

QWEN_MODEL_LICENSE_SHA256 = "8aff84e2629edb092f195c1d91bc368d8bb40eeb5fcedce2df19e405ee2cc876"
QWEN_NOTICE_BYTES = (
    b"Qwen is licensed under the Qwen LICENSE AGREEMENT, Copyright (c) "
    b"Alibaba Cloud. All Rights Reserved.\n"
)

POLICY_DIR = Path(__file__).resolve().parent
QWEN_NOTICE_PATH = POLICY_DIR / "QWEN_NOTICE.txt"
ATTRIBUTION_PLAN_PATH = POLICY_DIR / "attribution_plan.json"


class CodeLicense(StrEnum):
    MIT = "MIT"


class ModelLicense(StrEnum):
    QWEN_2024_09_19 = "Qwen LICENSE AGREEMENT (September 19, 2024)"


class OutputUseDisposition(StrEnum):
    HACKATHON_SYNTHETIC_COUNSEL_BEFORE_PRODUCTION = (
        "hackathon_synthetic_finance_counsel_before_production"
    )


class DistributionAttribution(StrEnum):
    BASE = "Built with Qwen"
    DERIVED = "Improved using Qwen"


EXPECTED_ATTRIBUTION_PLAN = {
    "base_model_attribution": DistributionAttribution.BASE.value,
    "derived_model_attribution": DistributionAttribution.DERIVED.value,
    "distribution_requirements": [
        "Include QWEN_NOTICE.txt with every distributed base snapshot or derived adapter.",
        "Display the applicable attribution in the model card and distribution README.",
        (
            "Display the applicable attribution in any user-facing product surface "
            "distributing the model."
        ),
    ],
    "model_license": ModelLicense.QWEN_2024_09_19.value,
    "placements": [
        "QWEN_NOTICE.txt",
        "model card",
        "distribution README",
        "user-facing product attribution",
    ],
    "repo_code_license": CodeLicense.MIT.value,
    "schema_version": "distillery.qwen72b_fallback.attribution_plan.v1",
}


class LicenseComplianceEvidence(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.license_evidence.v1"] = (
        "distillery.qwen72b_fallback.license_evidence.v1"
    )
    source: Literal[VerificationSource.LOCAL_BYTES] = VerificationSource.LOCAL_BYTES
    model_license: Literal[ModelLicense.QWEN_2024_09_19] = ModelLicense.QWEN_2024_09_19
    model_license_body_sha256: str = Field(pattern=SHA256_PATTERN)
    output_use_disposition: Literal[
        OutputUseDisposition.HACKATHON_SYNTHETIC_COUNSEL_BEFORE_PRODUCTION
    ] = OutputUseDisposition.HACKATHON_SYNTHETIC_COUNSEL_BEFORE_PRODUCTION
    qwen_notice_sha256: str = Field(pattern=SHA256_PATTERN)
    attribution_plan_sha256: str = Field(pattern=SHA256_PATTERN)
    base_attribution: Literal[DistributionAttribution.BASE] = DistributionAttribution.BASE
    derived_attribution: Literal[DistributionAttribution.DERIVED] = DistributionAttribution.DERIVED
    repo_code_license: Literal[CodeLicense.MIT] = CodeLicense.MIT
    repo_license_sha256: str = Field(pattern=SHA256_PATTERN)
    pyproject_sha256: str = Field(pattern=SHA256_PATTERN)
    commercial_100m_mau_requires_separate_grant: Literal[True] = True
    production_requires_counsel_review: Literal[True] = True


def verify_license_artifacts(repo_root: Path) -> LicenseComplianceEvidence:
    """Hash exact obligations and reject any attribution or code-license drift."""
    notice_bytes = QWEN_NOTICE_PATH.read_bytes()
    if notice_bytes != QWEN_NOTICE_BYTES:
        raise ValueError("QWEN_NOTICE.txt differs from the required Qwen notice")

    attribution_bytes = ATTRIBUTION_PLAN_PATH.read_bytes()
    attribution = json.loads(attribution_bytes)
    if attribution != EXPECTED_ATTRIBUTION_PLAN:
        raise ValueError("attribution_plan.json differs from the sealed obligation plan")

    pyproject_path = repo_root / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    if pyproject.get("project", {}).get("license") != {"text": CodeLicense.MIT.value}:
        raise ValueError("repo code license must remain exactly MIT in pyproject.toml")

    repo_license_path = repo_root / "LICENSE"
    if not repo_license_path.is_file():
        raise FileNotFoundError("repo MIT LICENSE file is missing")

    return LicenseComplianceEvidence.seal(
        model_license_body_sha256=QWEN_MODEL_LICENSE_SHA256,
        qwen_notice_sha256=sha256_bytes(notice_bytes),
        attribution_plan_sha256=sha256_bytes(attribution_bytes),
        repo_license_sha256=sha256_file(repo_license_path),
        pyproject_sha256=sha256_file(pyproject_path),
    )
