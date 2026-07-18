"""Exact pins for Qwen/Qwen2.5-72B-Instruct. No unset sentinels may pass."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from experiments.aws_smoke.pins import UNSET_SENTINELS
from experiments.qwen72b_fallback import FALLBACK_ROLE_NAME, TEACHER_ROLE_NAME

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REV_RE = re.compile(r"^[0-9a-f]{40}$")

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
REVISION = "495f39366efef23836d0cfae4fbe635880d2be31"
MODEL_CONFIG_SHA256 = "14ca217334fe0fd10148413592d68c99eeb33431ed89c1afa130fee560be2a29"
TOKENIZER_SHA256 = "8d9faaae11e51ab274be7eb785767c43a34f46b0c573632a6e2b8c0edbb90000"
CHAT_TEMPLATE_SHA256 = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
LICENSE_FILE_SHA256 = "8aff84e2629edb092f195c1d91bc368d8bb40eeb5fcedce2df19e405ee2cc876"
LICENSE_ID = "qwen-license-agreement-2024-09-19"
# Explicit hackathon disposition: Qwen license (not Apache-2.0). Commercial use
# above 100M MAU requires a separate Alibaba Cloud grant. Output-use for
# synthetic Distillery teaching/SFT is accepted for this hackathon with counsel
# follow-up before any production deployment.
LICENSE_DISPOSITION = (
    "qwen-license-agreement-2024-09-19; hackathon teaching+oracle-sft output-use "
    "accepted; counsel follow-up before production; 100M-MAU commercial grant gate"
)
OUTPUT_USE_DISPOSITION = (
    "synthetic-finance-only; teacher-for-tinyfable and oracle-sft-on-72b-base; "
    "no customer data; no production serving without counsel review"
)
SPECIAL_TOKEN_MAP: dict[str, int] = {
    "additional_special_tokens[0]": 151644,
    "additional_special_tokens[1]": 151645,
    "additional_special_tokens[2]": 151646,
    "additional_special_tokens[3]": 151647,
    "additional_special_tokens[4]": 151648,
    "additional_special_tokens[5]": 151649,
    "additional_special_tokens[6]": 151650,
    "additional_special_tokens[7]": 151651,
    "additional_special_tokens[8]": 151652,
    "additional_special_tokens[9]": 151653,
    "additional_special_tokens[10]": 151654,
    "additional_special_tokens[11]": 151655,
    "additional_special_tokens[12]": 151656,
    "eos_token": 151645,
    "pad_token": 151643,
}
DISTILLERY_BUCKET = "distillery-225989358036-us-east-1"
MODELS_PREFIX = f"s3://{DISTILLERY_BUCKET}/models"
SNAPSHOT_S3_URI = (
    f"{MODELS_PREFIX}/Qwen/Qwen2.5-72B-Instruct/{REVISION}/"
)
INVENTORY_PATH = Path(__file__).with_name("weight_inventory.json")


def _reject_unset(value: str, *, field_name: str) -> str:
    stripped = value.strip()
    if stripped in UNSET_SENTINELS or stripped.upper() in UNSET_SENTINELS:
        raise ValueError(f"{field_name} is unset or a forbidden placeholder: {value!r}")
    return stripped


@lru_cache(maxsize=1)
def load_weight_inventory() -> dict[str, Any]:
    payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    if payload.get("revision") != REVISION:
        raise ValueError("weight inventory revision does not match sealed pin")
    if payload.get("model_id") != MODEL_ID:
        raise ValueError("weight inventory model_id does not match sealed pin")
    if payload.get("tokenizer_sha256") != TOKENIZER_SHA256:
        raise ValueError("weight inventory tokenizer_sha256 mismatch")
    if payload.get("chat_template_sha256") != CHAT_TEMPLATE_SHA256:
        raise ValueError("weight inventory chat_template_sha256 mismatch")
    if payload.get("model_config_sha256") != MODEL_CONFIG_SHA256:
        raise ValueError("weight inventory model_config_sha256 mismatch")
    return payload


class Qwen72BIdentity(BaseModel):
    """Pinned identity shared by both teacher and fallback roles."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: Literal["Qwen/Qwen2.5-72B-Instruct"] = MODEL_ID
    revision: StrictStr = REVISION
    model_config_sha256: StrictStr = MODEL_CONFIG_SHA256
    tokenizer_sha256: StrictStr = TOKENIZER_SHA256
    chat_template_sha256: StrictStr = CHAT_TEMPLATE_SHA256
    license_file_sha256: StrictStr = LICENSE_FILE_SHA256
    license_id: StrictStr = LICENSE_ID
    license_disposition: StrictStr = LICENSE_DISPOSITION
    output_use_disposition: StrictStr = OUTPUT_USE_DISPOSITION
    special_token_map: dict[str, int] = Field(default_factory=lambda: dict(SPECIAL_TOKEN_MAP))
    snapshot_s3_uri: StrictStr = SNAPSHOT_S3_URI
    inventory_sha256: StrictStr

    @field_validator(
        "revision",
        "model_config_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "license_file_sha256",
        "license_id",
        "license_disposition",
        "output_use_disposition",
        "snapshot_s3_uri",
        "inventory_sha256",
        mode="before",
    )
    @classmethod
    def _no_placeholders(cls, value: object, info: object) -> object:
        if isinstance(value, str):
            field_name = getattr(info, "field_name", "field")
            return _reject_unset(value, field_name=str(field_name))
        return value

    @model_validator(mode="after")
    def _validate_identity(self) -> Qwen72BIdentity:
        if not _REV_RE.fullmatch(self.revision):
            raise ValueError("revision must be a 40-char lowercase git sha")
        for name in (
            "model_config_sha256",
            "tokenizer_sha256",
            "chat_template_sha256",
            "license_file_sha256",
            "inventory_sha256",
        ):
            if not _SHA256_RE.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be 64-char lowercase hex")
        if self.revision != REVISION:
            raise ValueError("revision drifted from sealed pin")
        if "qwen-license" not in self.license_disposition.lower():
            raise ValueError("license_disposition must name the Qwen license explicitly")
        if "apache-2.0" in self.license_disposition.lower() and "not apache" not in (
            self.license_disposition.lower()
        ):
            # 72B is NOT Apache-2.0; refuse a false Apache claim.
            raise ValueError(
                "Qwen2.5-72B-Instruct is under the Qwen LICENSE AGREEMENT, not Apache-2.0"
            )
        return self


def sealed_identity() -> Qwen72BIdentity:
    inventory = load_weight_inventory()
    return Qwen72BIdentity(inventory_sha256=str(inventory["inventory_sha256"]))


class RoleBinding(BaseModel):
    """One role bound to the same base snapshot with distinct scientific meaning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["teacher", "oracle_sft_adapted_fallback"]
    identity: Qwen72BIdentity
    supervision_source: StrictStr = Field(min_length=1)
    may_be_called_distilled_student: Literal[False] = False
    notes: StrictStr = Field(min_length=1)

    @model_validator(mode="after")
    def _role_rules(self) -> RoleBinding:
        if self.may_be_called_distilled_student:
            raise ValueError("72B adapted fallback must not be called a distilled student")
        if self.role == TEACHER_ROLE_NAME:
            if "tinyfable" not in self.supervision_source.lower():
                raise ValueError("teacher role must supervise TinyFable tiers")
        if self.role == FALLBACK_ROLE_NAME:
            source = self.supervision_source.lower()
            if "synthetic" not in source:
                raise ValueError("fallback role requires synthetic oracle/sequence supervision")
            # Allow explicit negations; reject affirmative larger-teacher supervision.
            if (
                "larger teacher" in source
                and "no separately identified larger teacher" not in source
                and "without a larger teacher" not in source
            ):
                raise ValueError(
                    "fallback role must not claim a larger teacher unless one is identified"
                )
        return self


def teacher_role_binding(identity: Qwen72BIdentity | None = None) -> RoleBinding:
    return RoleBinding(
        role="teacher",
        identity=identity or sealed_identity(),
        supervision_source=(
            "teacher_for_tinyfable_nano_core_plus; emits precomputed trajectories "
            "for smaller Qwen2.5 students"
        ),
        notes=(
            "Same pinned 72B snapshot acts as a powerful teacher. TinyFable tiers "
            "are the students. The 72B weights themselves are not a student here."
        ),
    )


def fallback_role_binding(identity: Qwen72BIdentity | None = None) -> RoleBinding:
    return RoleBinding(
        role="oracle_sft_adapted_fallback",
        identity=identity or sealed_identity(),
        supervision_source=(
            "synthetic_finance_oracle_and_sequence_sft_on_72b_base; "
            "no separately identified larger teacher"
        ),
        notes=(
            "Post-trains the 72B base into an expensive quality finance fallback. "
            "TinyFable remains the deployable small model. Do not call this adapted "
            "72B a distilled student."
        ),
    )
