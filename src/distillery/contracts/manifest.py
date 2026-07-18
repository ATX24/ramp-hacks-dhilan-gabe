"""Sealed, SHA-256-addressed run manifest (distillery.run.v1)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictStr, field_validator, model_validator

from distillery.contracts.base import (
    FrozenDict,
    FrozenModel,
    deep_thaw,
)
from distillery.contracts.budgets import WEIGHT_SUM_TOLERANCE
from distillery.contracts.hashing import (
    AwareDatetime,
    GitCommitSha,
    NonNegativeSafeInt,
    PositiveSafeInt,
    PrefixedSha256,
    Sha256Hex,
    content_sha256,
)
from distillery.contracts.ids import DatasetId, ExampleId, RunId
from distillery.contracts.recipes import (
    AutoResolverInput,
    RequestedRecipe,
    ResolvedRecipe,
    resolve_auto_recipe,
    validate_recipe_resolution,
)
from distillery.contracts.tasks import LabelSource, SplitName

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


def manifest_memory_dry_run_evidence_sha256(
    evidence: Mapping[str, object],
) -> str:
    """Hash all memory-probe fields except its self-referential digest."""
    payload = deep_thaw(evidence)
    payload.pop("evidence_sha256", None)
    return content_sha256(payload)


class ManifestDatasetRef(FrozenModel):
    dataset_id: DatasetId
    uri: StrictStr = Field(min_length=1)
    sha256: Sha256Hex
    split_sha256: FrozenDict[SplitName, Sha256Hex]

    @field_validator("split_sha256")
    @classmethod
    def _required_splits(
        cls,
        value: Mapping[SplitName, Sha256Hex],
    ) -> Mapping[SplitName, Sha256Hex]:
        required = {SplitName.TRAIN, SplitName.VALIDATION}
        missing = required - set(value)
        if missing:
            names = ", ".join(sorted(split.value for split in missing))
            raise ValueError(f"manifest split_sha256 missing required splits: {names}")
        return value


class ManifestModelSpec(FrozenModel):
    id: StrictStr = Field(min_length=1)
    revision: GitCommitSha
    tokenizer_sha256: Sha256Hex
    chat_template_sha256: Sha256Hex


class ManifestModels(FrozenModel):
    teacher: ManifestModelSpec
    student: ManifestModelSpec


class ManifestRecipe(FrozenModel):
    requested: RequestedRecipe
    resolved: ResolvedRecipe
    resolver_reasons: tuple[StrictStr, ...] = ()

    @model_validator(mode="after")
    def _no_silent_downgrade(self) -> ManifestRecipe:
        validate_recipe_resolution(
            self.requested,
            self.resolved,
            self.resolver_reasons,
        )
        return self


class ManifestSpecialTokenMapEvidence(FrozenModel):
    """Exact token-name-to-ID evidence consumed by the logit adapter."""

    teacher: FrozenDict[StrictStr, NonNegativeSafeInt] = Field(min_length=1)
    student: FrozenDict[StrictStr, NonNegativeSafeInt] = Field(min_length=1)

    @model_validator(mode="after")
    def _maps_match_exactly(self) -> ManifestSpecialTokenMapEvidence:
        if self.teacher != self.student:
            raise ValueError("teacher and student special-token maps must match exactly")
        return self


class ManifestMemoryDryRunEvidence(FrozenModel):
    """Precomputed logit memory probe bound to the exact sealed configuration."""

    schema_version: Literal["distillery.memory_dry_run.v2"] = "distillery.memory_dry_run.v2"
    passed: StrictBool
    binding_sha256: Sha256Hex
    evidence_sha256: Sha256Hex
    training_config_sha256: Sha256Hex
    teacher_model_config_sha256: Sha256Hex
    student_model_config_sha256: Sha256Hex
    length_config_sha256: Sha256Hex
    runtime_image_digest: PrefixedSha256
    instance_type: StrictStr = Field(min_length=1)
    recipe_id: Literal["logit.v1"] = "logit.v1"
    teacher_model_id: StrictStr = Field(min_length=1)
    teacher_revision: GitCommitSha
    student_model_id: StrictStr = Field(min_length=1)
    student_revision: GitCommitSha
    max_length: PositiveSafeInt
    max_completion: PositiveSafeInt
    vocab_chunk_size: PositiveSafeInt
    peak_memory_bytes: PositiveSafeInt
    capacity_memory_bytes: PositiveSafeInt
    headroom_bytes: PositiveSafeInt
    device_type: StrictStr = Field(min_length=1)
    probe_id: StrictStr = Field(min_length=1)

    @model_validator(mode="after")
    def _bound_memory_measurement(self) -> ManifestMemoryDryRunEvidence:
        if self.peak_memory_bytes >= self.capacity_memory_bytes:
            raise ValueError("peak memory must be below measured capacity")
        if self.headroom_bytes != (self.capacity_memory_bytes - self.peak_memory_bytes):
            raise ValueError("headroom_bytes must equal capacity minus peak")
        if self.evidence_sha256 != manifest_memory_dry_run_evidence_sha256(
            self.model_dump(mode="json")
        ):
            raise ValueError("evidence_sha256 must bind the complete memory evidence record")
        return self


class ManifestTrainingCapabilityEvidence(FrozenModel):
    """Typed evidence at the adapter-compatible sealed wire location."""

    schema_version: Literal["distillery.training_capabilities.v1"] = (
        "distillery.training_capabilities.v1"
    )
    special_token_maps: ManifestSpecialTokenMapEvidence | None = None
    memory_dry_run: ManifestMemoryDryRunEvidence | None = None
    auto_resolver_input: AutoResolverInput | None = None


class ManifestCompletionEvidence(FrozenModel):
    """Student-tokenizer completion counts and response provenance seal."""

    schema_version: Literal["distillery.completion_evidence.v1"] = (
        "distillery.completion_evidence.v1"
    )
    source_file_sha256: Sha256Hex
    canonical_records_sha256: Sha256Hex
    record_sha256: FrozenDict[ExampleId, Sha256Hex] = Field(min_length=1)
    provenance_sha256: Sha256Hex
    completion_token_counts: FrozenDict[ExampleId, PositiveSafeInt] = Field(min_length=1)
    completion_token_count_source: Literal["student_tokenizer"] = "student_tokenizer"
    completion_tokenizer_sha256: Sha256Hex
    label_source_counts: FrozenDict[LabelSource, NonNegativeSafeInt] = Field(min_length=1)
    accepted_example_count: PositiveSafeInt

    @model_validator(mode="after")
    def _count_totals_match(self) -> ManifestCompletionEvidence:
        if len(self.completion_token_counts) != self.accepted_example_count:
            raise ValueError("accepted_example_count must equal the number of completion counts")
        if set(self.record_sha256) != set(self.completion_token_counts):
            raise ValueError(
                "record_sha256 and completion_token_counts must cover identical examples"
            )
        if sum(self.label_source_counts.values()) != self.accepted_example_count:
            raise ValueError("label_source_counts must sum to accepted_example_count")
        return self


class ManifestQLoRAConfig(FrozenModel, Mapping[str, object]):
    """Strict QLoRA/objective config with typed capability evidence."""

    rank: PositiveSafeInt = 8
    alpha: PositiveSafeInt = 16
    dropout: FiniteFloat = Field(default=0.05, ge=0.0, le=1.0)
    target_modules: tuple[StrictStr, ...] = Field(
        default=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        min_length=1,
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: StrictStr = Field(default="CAUSAL_LM", min_length=1)
    use_rslora: StrictBool = False
    modules_to_save: tuple[StrictStr, ...] = ()
    max_completion: PositiveSafeInt = 160
    logit_temperature: FiniteFloat = Field(default=2.0, gt=0.0)
    kd_weight: FiniteFloat = Field(default=0.7, ge=0.0, le=1.0)
    hard_ce_weight: FiniteFloat = Field(default=0.3, ge=0.0, le=1.0)
    vocab_chunk: PositiveSafeInt = 4096
    capability_evidence: ManifestTrainingCapabilityEvidence | None = None

    @model_validator(mode="after")
    def _objective_weights_sum_to_one(self) -> ManifestQLoRAConfig:
        total = self.kd_weight + self.hard_ce_weight
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"kd_weight + hard_ce_weight must equal 1.0, got {total}")
        return self

    def __getitem__(self, key: str) -> object:
        if key not in type(self).model_fields:
            raise KeyError(key)
        return deep_thaw(getattr(self, key))

    def __iter__(self) -> Iterator[str]:
        return iter(type(self).model_fields)

    def __len__(self) -> int:
        return len(type(self).model_fields)


class ManifestTraining(FrozenModel):
    seed: NonNegativeSafeInt
    max_steps: PositiveSafeInt
    token_budget: NonNegativeSafeInt
    max_length: PositiveSafeInt
    qlora: ManifestQLoRAConfig = Field(default_factory=ManifestQLoRAConfig)
    completion_evidence: ManifestCompletionEvidence | None = None

    @model_validator(mode="after")
    def _completion_bounds(self) -> ManifestTraining:
        if self.qlora.max_completion > self.max_length:
            raise ValueError("qlora.max_completion cannot exceed max_length")
        if (
            self.token_budget > 0
            and self.completion_evidence is not None
            and sum(self.completion_evidence.completion_token_counts.values()) > self.token_budget
        ):
            raise ValueError("completion token counts exceed token_budget")
        return self


class ManifestProofProtocol(FrozenModel):
    id: StrictStr = Field(min_length=1)
    sha256: Sha256Hex


class ManifestRuntime(FrozenModel):
    backend: Literal["local", "sagemaker"]
    region: StrictStr = Field(min_length=1)
    instance_type: StrictStr = Field(min_length=1)
    image_digest: PrefixedSha256


class ManifestCost(FrozenModel):
    max_run_usd: FiniteFloat = Field(gt=0)
    estimate_low_usd: FiniteFloat = Field(ge=0)
    estimate_high_usd: FiniteFloat = Field(ge=0)

    @model_validator(mode="after")
    def _estimate_range_within_ceiling(self) -> ManifestCost:
        if self.estimate_low_usd > self.estimate_high_usd:
            raise ValueError("estimate_low_usd cannot exceed estimate_high_usd")
        if self.estimate_high_usd > self.max_run_usd:
            raise ValueError("estimate_high_usd cannot exceed max_run_usd")
        return self


class ManifestOutput(FrozenModel):
    prefix: StrictStr = Field(min_length=1)


def _manifest_payload(value: SealedRunManifest | Mapping[str, object]) -> dict[str, object]:
    if isinstance(value, FrozenModel):
        payload = value.model_dump(mode="json")
    else:
        payload = deep_thaw(value)
    if not isinstance(payload, dict):
        raise TypeError("manifest payload must be a JSON object")
    return payload


def _payload_without_capability_evidence(
    value: SealedRunManifest | Mapping[str, object],
) -> dict[str, object]:
    payload = _manifest_payload(value)
    training = dict(payload["training"])
    qlora = dict(training["qlora"])
    qlora.pop("capability_evidence", None)
    training["qlora"] = qlora
    payload["training"] = training
    return payload


def manifest_training_configuration_sha256(
    value: SealedRunManifest | Mapping[str, object],
) -> str:
    """Hash training config exactly as the current training adapter does."""
    payload = _payload_without_capability_evidence(value)
    return content_sha256(payload["training"])


def manifest_model_configuration_sha256(model_spec: ManifestModelSpec) -> str:
    """Hash model identity plus tokenizer/chat-template configuration."""
    return content_sha256(model_spec.model_dump(mode="json"))


def manifest_length_configuration_sha256(manifest: SealedRunManifest) -> str:
    """Hash exact sequence and vocabulary-chunk bounds used by logit.v1."""
    return content_sha256(
        {
            "max_length": manifest.training.max_length,
            "max_completion": manifest.training.qlora.max_completion,
            "vocab_chunk_size": manifest.training.qlora.vocab_chunk,
        }
    )


def manifest_capability_binding_sha256(
    value: SealedRunManifest | Mapping[str, object],
) -> str:
    """Bind capability evidence to manifest, token maps, resolver input, and image."""
    payload = _manifest_payload(value)
    training = dict(payload["training"])
    qlora = dict(training["qlora"])
    raw_evidence = qlora.get("capability_evidence")
    if not isinstance(raw_evidence, dict):
        raise ValueError("capability_evidence is required to compute its binding")
    special_token_maps = raw_evidence.get("special_token_maps")
    if not isinstance(special_token_maps, dict):
        raise ValueError("special_token_maps are required to compute capability binding")
    return content_sha256(
        {
            "manifest_without_capability_evidence": (_payload_without_capability_evidence(payload)),
            "special_token_maps": special_token_maps,
            "auto_resolver_input": raw_evidence.get("auto_resolver_input"),
        }
    )


class SealedRunManifest(FrozenModel):
    """Once sealed, any change requires a new run_id."""

    schema_version: Literal["distillery.run.v1"] = "distillery.run.v1"
    run_id: RunId
    created_at: AwareDatetime
    dataset: ManifestDatasetRef
    models: ManifestModels
    recipe: ManifestRecipe
    training: ManifestTraining
    proof_protocol: ManifestProofProtocol
    runtime: ManifestRuntime
    cost: ManifestCost
    output: ManifestOutput
    package_lock_hash: Sha256Hex
    source_revision: StrictStr = Field(min_length=1)
    license_dispositions: FrozenDict[StrictStr, StrictStr] = Field(default_factory=dict)
    tags: FrozenDict[StrictStr, StrictStr] = Field(default_factory=dict)
    sampler_order_hash: Sha256Hex

    @model_validator(mode="after")
    def _cross_validate_sealed_evidence(self) -> SealedRunManifest:
        evidence = self.training.qlora.capability_evidence
        auto_input = evidence.auto_resolver_input if evidence is not None else None

        if self.recipe.requested == "auto":
            if auto_input is None:
                raise ValueError("auto recipes require sealed auto_resolver_input")
            recomputed = resolve_auto_recipe(auto_input)
            if recomputed.resolved is None:
                raise ValueError("sealed auto resolver input does not resolve a recipe")
            if (
                recomputed.resolved != self.recipe.resolved
                or recomputed.reasons != self.recipe.resolver_reasons
            ):
                raise ValueError(
                    "sealed auto resolver input/reasons do not match recipe resolution"
                )

        if self.recipe.resolved == "do_not_distill":
            if self.training.completion_evidence is not None:
                raise ValueError("do_not_distill cannot seal training completion evidence")
            if evidence is not None and (
                evidence.special_token_maps is not None or evidence.memory_dry_run is not None
            ):
                raise ValueError("do_not_distill cannot carry logit training evidence")
            return self

        completion = self.training.completion_evidence
        if completion is None:
            raise ValueError("trainable recipes require sealed completion evidence")
        if completion.completion_tokenizer_sha256 != self.models.student.tokenizer_sha256:
            raise ValueError("completion evidence must be bound to the student tokenizer")

        if self.recipe.resolved != "logit.v1":
            return self

        if evidence is None:
            raise ValueError("logit.v1 requires complete capability evidence")
        if auto_input is None:
            raise ValueError("logit.v1 requires complete capability assertions")
        required_capabilities = (
            auto_input.local_white_box,
            auto_input.tokenizer_fingerprint_match,
            auto_input.special_token_map_match,
            auto_input.chat_template_compatible,
            auto_input.memory_dry_run_ok,
        )
        if any(value is not True for value in required_capabilities):
            raise ValueError("unknown or failed logit.v1 capability is not success")
        if self.models.teacher.tokenizer_sha256 != self.models.student.tokenizer_sha256:
            raise ValueError("logit.v1 requires equal tokenizer fingerprints")
        if self.models.teacher.chat_template_sha256 != self.models.student.chat_template_sha256:
            raise ValueError("logit.v1 requires equal chat-template fingerprints")
        if evidence.special_token_maps is None:
            raise ValueError("logit.v1 requires nonempty special-token map evidence")
        memory = evidence.memory_dry_run
        if memory is None:
            raise ValueError("logit.v1 requires memory dry-run evidence")
        expected_fields = {
            "teacher_model_id": self.models.teacher.id,
            "teacher_revision": self.models.teacher.revision,
            "student_model_id": self.models.student.id,
            "student_revision": self.models.student.revision,
            "max_length": self.training.max_length,
            "max_completion": self.training.qlora.max_completion,
            "vocab_chunk_size": self.training.qlora.vocab_chunk,
            "training_config_sha256": manifest_training_configuration_sha256(self),
            "binding_sha256": manifest_capability_binding_sha256(self),
            "teacher_model_config_sha256": (
                manifest_model_configuration_sha256(self.models.teacher)
            ),
            "student_model_config_sha256": (
                manifest_model_configuration_sha256(self.models.student)
            ),
            "length_config_sha256": manifest_length_configuration_sha256(self),
            "runtime_image_digest": self.runtime.image_digest,
            "instance_type": self.runtime.instance_type,
        }
        mismatches = {
            name: {"expected": expected, "actual": getattr(memory, name)}
            for name, expected in expected_fields.items()
            if getattr(memory, name) != expected
        }
        if not memory.passed:
            mismatches["passed"] = {"expected": True, "actual": False}
        if mismatches:
            raise ValueError(f"logit.v1 memory dry-run evidence binding mismatch: {mismatches}")
        return self

    def seal_sha256(self) -> str:
        """Content address of the fully frozen manifest."""
        return content_sha256(self)
