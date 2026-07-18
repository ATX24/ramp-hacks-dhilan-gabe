"""Exact Qwen2.5-72B identity pins and self-verifying weight inventory."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from distillery.contracts.hashing import content_sha256
from experiments.qwen72b_fallback.evidence import (
    REVISION_PATTERN,
    SHA256_PATTERN,
    HashBoundEvidence,
    VerificationSource,
    sha256_bytes,
)
from experiments.qwen72b_fallback.license_policy import QWEN_MODEL_LICENSE_SHA256

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
REVISION = "495f39366efef23836d0cfae4fbe635880d2be31"
MODEL_CONFIG_SHA256 = "14ca217334fe0fd10148413592d68c99eeb33431ed89c1afa130fee560be2a29"
TOKENIZER_SHA256 = "8d9faaae11e51ab274be7eb785767c43a34f46b0c573632a6e2b8c0edbb90000"
CHAT_TEMPLATE_SHA256 = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
LICENSE_FILE_SHA256 = QWEN_MODEL_LICENSE_SHA256

DISTILLERY_ACCOUNT_ID = "225989358036"
AWS_REGION = "us-east-1"
DISTILLERY_BUCKET = f"distillery-{DISTILLERY_ACCOUNT_ID}-{AWS_REGION}"
MODELS_PREFIX = f"s3://{DISTILLERY_BUCKET}/models"
SNAPSHOT_S3_URI = f"{MODELS_PREFIX}/Qwen/Qwen2.5-72B-Instruct/{REVISION}/"
ECR_REPOSITORY = "distillery-training"

PACKAGE_DIR = Path(__file__).resolve().parent
INVENTORY_PATH = PACKAGE_DIR / "weight_inventory.json"
EXECUTION_BINDINGS_PATH = PACKAGE_DIR / "execution_bindings.json"
TOKENIZER_TARGETS_PATH = PACKAGE_DIR / "tokenizer_targets.json"

TOKENIZER_FILE_SHA256 = {
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": ("5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"),
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
SPECIAL_TOKEN_IDS = {
    "<|box_end|>": 151649,
    "<|box_start|>": 151648,
    "<|endoftext|>": 151643,
    "<|image_pad|>": 151655,
    "<|im_end|>": 151645,
    "<|im_start|>": 151644,
    "<|object_ref_end|>": 151647,
    "<|object_ref_start|>": 151646,
    "<|quad_end|>": 151651,
    "<|quad_start|>": 151650,
    "<|video_pad|>": 151656,
    "<|vision_end|>": 151653,
    "<|vision_pad|>": 151654,
    "<|vision_start|>": 151652,
}


class InventoryFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    sha256: str = Field(pattern=SHA256_PATTERN)
    size: int = Field(gt=0)


class WeightInventory(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["distillery.qwen72b_fallback.weight_inventory.v1"]
    model_id: Literal["Qwen/Qwen2.5-72B-Instruct"]
    revision: str = Field(pattern=REVISION_PATTERN)
    hf_source: str
    architecture: Literal["Qwen2ForCausalLM"]
    model_type: Literal["qwen2"]
    hidden_size: Literal[8192]
    num_hidden_layers: Literal[80]
    num_attention_heads: Literal[64]
    num_key_value_heads: Literal[8]
    vocab_size: Literal[152064]
    torch_dtype: Literal["bfloat16"]
    n_safetensors_shards: Literal[37]
    total_safetensors_bytes: int = Field(gt=0)
    files: dict[str, InventoryFile]
    tokenizer_files_sha256: dict[str, str]
    tokenizer_sha256: str = Field(pattern=SHA256_PATTERN)
    chat_template_sha256: str = Field(pattern=SHA256_PATTERN)
    special_token_ids: dict[str, int]
    model_config_sha256: str = Field(pattern=SHA256_PATTERN)
    license_file_sha256: str = Field(pattern=SHA256_PATTERN)
    license_id: Literal["qwen-license-agreement-2024-09-19"]
    family_reference_models: tuple[str, ...]
    inventory_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _verify_inventory(self) -> WeightInventory:
        body = self.model_dump(mode="json", exclude={"inventory_sha256"})
        if self.inventory_sha256 != content_sha256(body):
            raise ValueError("weight inventory canonical hash mismatch")
        if self.model_id != MODEL_ID or self.revision != REVISION:
            raise ValueError("weight inventory model identity differs from exact pin")
        if self.model_config_sha256 != MODEL_CONFIG_SHA256:
            raise ValueError("weight inventory config hash differs from exact pin")
        if self.tokenizer_sha256 != TOKENIZER_SHA256:
            raise ValueError("weight inventory tokenizer aggregate hash differs from exact pin")
        if self.chat_template_sha256 != CHAT_TEMPLATE_SHA256:
            raise ValueError("weight inventory chat-template hash differs from exact pin")
        if self.license_file_sha256 != LICENSE_FILE_SHA256:
            raise ValueError("weight inventory license hash differs from exact pin")
        if self.tokenizer_files_sha256 != TOKENIZER_FILE_SHA256:
            raise ValueError("weight inventory tokenizer file hashes differ from exact pins")
        if self.special_token_ids != SPECIAL_TOKEN_IDS:
            raise ValueError("weight inventory special token IDs differ from exact pins")
        shards = {name: item for name, item in self.files.items() if name.endswith(".safetensors")}
        expected_names = {f"model-{index:05d}-of-00037.safetensors" for index in range(1, 38)}
        if set(shards) != expected_names:
            raise ValueError("weight inventory must contain exactly all 37 shard names")
        if sum(item.size for item in shards.values()) != self.total_safetensors_bytes:
            raise ValueError("weight shard byte total differs from sealed total")
        required_small = {
            "LICENSE",
            "README.md",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        }
        if not required_small <= set(self.files):
            raise ValueError("weight inventory lacks required model/tokenizer/license files")
        for name, digest in TOKENIZER_FILE_SHA256.items():
            if self.files[name].sha256 != digest:
                raise ValueError(f"weight inventory {name} body hash differs from exact pin")
        return self


class Qwen72BIdentityEvidence(HashBoundEvidence):
    schema_version: Literal["distillery.qwen72b_fallback.identity_evidence.v2"] = (
        "distillery.qwen72b_fallback.identity_evidence.v2"
    )
    source: Literal[VerificationSource.LOCAL_BYTES] = VerificationSource.LOCAL_BYTES
    model_id: Literal["Qwen/Qwen2.5-72B-Instruct"] = MODEL_ID
    revision: str = Field(pattern=REVISION_PATTERN)
    model_config_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_file_sha256: dict[str, str]
    chat_template_sha256: str = Field(pattern=SHA256_PATTERN)
    special_token_ids: dict[str, int]
    license_file_sha256: str = Field(pattern=SHA256_PATTERN)
    inventory_sha256: str = Field(pattern=SHA256_PATTERN)
    inventory_bytes_sha256: str = Field(pattern=SHA256_PATTERN)
    snapshot_s3_uri: Literal[
        "s3://distillery-225989358036-us-east-1/models/Qwen/"
        "Qwen2.5-72B-Instruct/495f39366efef23836d0cfae4fbe635880d2be31/"
    ] = SNAPSHOT_S3_URI


@lru_cache(maxsize=1)
def load_weight_inventory() -> WeightInventory:
    return WeightInventory.model_validate_json(INVENTORY_PATH.read_bytes())


@lru_cache(maxsize=1)
def sealed_identity() -> Qwen72BIdentityEvidence:
    inventory_bytes = INVENTORY_PATH.read_bytes()
    inventory = WeightInventory.model_validate_json(inventory_bytes)
    return Qwen72BIdentityEvidence.seal(
        revision=inventory.revision,
        model_config_sha256=inventory.model_config_sha256,
        tokenizer_sha256=inventory.tokenizer_sha256,
        tokenizer_file_sha256=dict(inventory.tokenizer_files_sha256),
        chat_template_sha256=inventory.chat_template_sha256,
        special_token_ids=dict(inventory.special_token_ids),
        license_file_sha256=inventory.license_file_sha256,
        inventory_sha256=inventory.inventory_sha256,
        inventory_bytes_sha256=sha256_bytes(inventory_bytes),
    )
