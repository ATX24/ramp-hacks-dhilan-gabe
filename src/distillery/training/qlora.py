"""QLoRA configuration builders (no peft/bitsandbytes instantiation)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from distillery.contracts.budgets import SmokeTrainingBudget, TrainingBudget
from distillery.contracts.errors import DistilleryError, DistilleryErrorCode, ErrorPayload

# Default projection modules for Qwen-style transformer blocks.
DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class QLoRAConfig(BaseModel):
    """Serializable QLoRA / LoRA adapter configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    rank: int = Field(ge=1)
    alpha: int = Field(ge=1)
    dropout: float = Field(ge=0.0, le=1.0)
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    modules_to_save: tuple[str, ...] = ()

    @field_validator("target_modules", "modules_to_save", mode="before")
    @classmethod
    def _strict_module_names(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise ValueError("module names must be supplied as a list or tuple")
        if any(
            not isinstance(module, str) or not module.strip() for module in value
        ):
            raise ValueError("module names must be nonempty strings")
        return tuple(value)

    @model_validator(mode="after")
    def _validate_modules(self) -> QLoRAConfig:
        if not self.target_modules:
            raise ValueError("target_modules must be nonempty")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("target_modules must not contain duplicates")
        if len(set(self.modules_to_save)) != len(self.modules_to_save):
            raise ValueError("modules_to_save must not contain duplicates")
        if not self.task_type.strip():
            raise ValueError("task_type must be nonempty")
        return self

    def to_peft_dict(self) -> dict[str, Any]:
        """Dict shaped for ``LoraConfig`` construction (caller imports peft)."""
        return {
            "r": self.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "bias": self.bias,
            "task_type": self.task_type,
            "use_rslora": self.use_rslora,
            "modules_to_save": list(self.modules_to_save),
        }


class QuantizationConfigSpec(BaseModel):
    """4-bit NF4 double-quantization settings for student base load."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"

    def to_bitsandbytes_dict(self) -> dict[str, Any]:
        return {
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
        }


def qlora_from_smoke_budget(budget: SmokeTrainingBudget | None = None) -> QLoRAConfig:
    b = budget or SmokeTrainingBudget()
    return QLoRAConfig(rank=b.lora_rank, alpha=b.lora_alpha, dropout=b.lora_dropout)


def qlora_from_training_budget(budget: TrainingBudget | None = None) -> QLoRAConfig:
    b = budget or TrainingBudget()
    return QLoRAConfig(rank=b.lora_rank, alpha=b.lora_alpha, dropout=b.lora_dropout)


def qlora_from_manifest_dict(qlora: dict[str, Any]) -> QLoRAConfig:
    """Strictly parse QLoRA fields and return typed Distillery errors."""
    if not isinstance(qlora, dict):
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.RECIPE_INCOMPATIBLE,
                "training.qlora must be an object",
            )
        )
    qlora_fields = {
        "rank",
        "alpha",
        "dropout",
        "target_modules",
        "bias",
        "task_type",
        "use_rslora",
        "modules_to_save",
    }
    trainer_adapter_fields = {
        "logit_temperature",
        "kd_weight",
        "hard_ce_weight",
        "vocab_chunk",
        "max_completion",
        "capability_evidence",
    }
    unknown = sorted(set(qlora) - qlora_fields - trainer_adapter_fields)
    if unknown:
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.RECIPE_INCOMPATIBLE,
                "training.qlora contains unsupported fields",
                details={"unknown_fields": unknown},
            )
        )
    smoke = SmokeTrainingBudget()
    try:
        return QLoRAConfig(
            rank=qlora.get("rank", smoke.lora_rank),
            alpha=qlora.get("alpha", smoke.lora_alpha),
            dropout=qlora.get("dropout", smoke.lora_dropout),
            target_modules=qlora.get(
                "target_modules", DEFAULT_LORA_TARGET_MODULES
            ),
            bias=qlora.get("bias", "none"),
            task_type=qlora.get("task_type", "CAUSAL_LM"),
            use_rslora=qlora.get("use_rslora", False),
            modules_to_save=qlora.get("modules_to_save", ()),
        )
    except (ValidationError, ValueError, TypeError) as exc:
        details: dict[str, Any]
        if isinstance(exc, ValidationError):
            details = {
                "validation_errors": [
                    {
                        "type": error["type"],
                        "loc": list(error["loc"]),
                        "msg": error["msg"],
                        "input": repr(error.get("input")),
                    }
                    for error in exc.errors(include_url=False)
                ]
            }
        else:
            details = {"reason": str(exc)}
        raise DistilleryError(
            ErrorPayload.from_code(
                DistilleryErrorCode.RECIPE_INCOMPATIBLE,
                "training.qlora contains malformed values",
                details=details,
            )
        ) from None


def default_student_quantization() -> QuantizationConfigSpec:
    return QuantizationConfigSpec()
