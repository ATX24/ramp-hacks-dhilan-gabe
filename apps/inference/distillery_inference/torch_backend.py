"""Torch/PEFT backend loaded only when DISTILLERY_INFERENCE_RUNTIME=torch."""

from __future__ import annotations

import threading
import time
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from distillery_inference.bundle import ArtifactManifest, LoadedBundle
from distillery_inference.errors import InferenceError, InferenceErrorCode
from distillery_inference.prompts import build_messages, render_chat_prompt
from distillery_inference.runtime import GenerationResult
from distillery_inference.schemas import FinanceTaskId


class TorchBackend:
    """One pinned local base + concurrency-safe PEFT adapter switching."""

    def __init__(self, bundle: LoadedBundle) -> None:
        self._bundle = bundle
        self._lock = threading.RLock()
        self._loaded_model_id: str | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._base_model: Any = None
        self._load_base()

    def ready(self) -> bool:
        with self._lock:
            return self._base_model is not None and self._tokenizer is not None

    def loaded_model_id(self) -> str | None:
        with self._lock:
            return self._loaded_model_id

    def available_model_ids(self) -> list[str]:
        return sorted(self._bundle.artifacts_by_model)

    def ensure_artifact(self, artifact: ArtifactManifest) -> None:
        with self._lock:
            if self._loaded_model_id == artifact.model_id and self._model is not None:
                return
            try:
                self._switch_locked(artifact)
            except InferenceError:
                raise
            except Exception as exc:
                raise InferenceError(
                    InferenceErrorCode.ADAPTER_SWITCH_FAILED,
                    f"Failed to load artifact {artifact.artifact_id}: {exc}",
                    http_status=500,
                    retryable=True,
                    details={"model_id": artifact.model_id},
                ) from exc

    def generate(
        self,
        *,
        artifact: ArtifactManifest,
        task: FinanceTaskId,
        example_input: dict[str, Any],
        max_prompt_tokens: int,
        max_completion_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        timeout_s: float,
    ) -> GenerationResult:
        started = time.monotonic()
        with self._lock:
            self.ensure_artifact(artifact)
            messages = build_messages(task=task, example_input=example_input)
            if hasattr(self._tokenizer, "apply_chat_template"):
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = render_chat_prompt(messages)
            encoded = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=False,
                add_special_tokens=False,
            )
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            if prompt_tokens > max_prompt_tokens:
                raise InferenceError(
                    InferenceErrorCode.TOKEN_LIMIT_EXCEEDED,
                    (
                        f"Prompt tokens {prompt_tokens} exceed limit "
                        f"{max_prompt_tokens}"
                    ),
                    http_status=413,
                    details={
                        "prompt_tokens": prompt_tokens,
                        "max_prompt_tokens": max_prompt_tokens,
                    },
                )
            if time.monotonic() - started > timeout_s:
                raise InferenceError(
                    InferenceErrorCode.TIMEOUT,
                    f"Inference timed out after {timeout_s:.1f}s",
                    http_status=504,
                    retryable=True,
                )
            input_ids = encoded["input_ids"].to(self._model.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._model.device)
            torch.manual_seed(seed)
            generate_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "max_new_tokens": max_completion_tokens,
                "do_sample": temperature > 0,
                "temperature": max(temperature, 1e-5),
                "top_p": top_p,
                "pad_token_id": self._tokenizer.pad_token_id
                or self._tokenizer.eos_token_id,
            }
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask
            with torch.inference_mode():
                output_ids = self._model.generate(**generate_kwargs)
            if time.monotonic() - started > timeout_s:
                raise InferenceError(
                    InferenceErrorCode.TIMEOUT,
                    f"Inference timed out after {timeout_s:.1f}s",
                    http_status=504,
                    retryable=True,
                )
            new_tokens = output_ids[0, prompt_tokens:]
            completion_tokens = int(new_tokens.shape[-1])
            if completion_tokens > max_completion_tokens:
                raise InferenceError(
                    InferenceErrorCode.TOKEN_LIMIT_EXCEEDED,
                    (
                        f"Completion tokens {completion_tokens} exceed limit "
                        f"{max_completion_tokens}"
                    ),
                    http_status=413,
                )
            raw_output = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            return GenerationResult(
                raw_output=raw_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model_id=artifact.model_id,
                artifact_id=artifact.artifact_id,
            )

    def _load_base(self) -> None:
        base_path = self._bundle.base_path()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(base_path),
                local_files_only=True,
                revision=self._bundle.registry.tokenizer_revision,
            )
            self._base_model = AutoModelForCausalLM.from_pretrained(
                str(base_path),
                local_files_only=True,
                revision=self._bundle.registry.base_revision,
                torch_dtype=torch.float32,
            )
            self._base_model.eval()
            self._model = self._base_model
        except Exception as exc:
            raise InferenceError(
                InferenceErrorCode.SERVING_NOT_READY,
                f"Failed to load pinned base model from bundle: {exc}",
                http_status=503,
            ) from exc

    def _switch_locked(self, artifact: ArtifactManifest) -> None:
        if self._base_model is None or self._tokenizer is None:
            raise InferenceError(
                InferenceErrorCode.SERVING_NOT_READY,
                "Base model is not loaded",
                http_status=503,
            )
        artifact_path = self._bundle.artifact_path(artifact)
        if artifact.kind == "base":
            self._model = self._base_model
            self._loaded_model_id = artifact.model_id
            return
        if artifact.kind == "merged":
            self._model = AutoModelForCausalLM.from_pretrained(
                str(artifact_path),
                local_files_only=True,
                torch_dtype=torch.float32,
            )
            self._model.eval()
            self._loaded_model_id = artifact.model_id
            return
        if artifact.kind == "peft_adapter":
            # Always re-wrap from the immutable base; never chain adapters.
            self._model = PeftModel.from_pretrained(
                self._base_model,
                str(artifact_path),
                local_files_only=True,
            )
            self._model.eval()
            self._loaded_model_id = artifact.model_id
            return
        raise InferenceError(
            InferenceErrorCode.ARTIFACT_NOT_SERVABLE,
            f"Unsupported artifact kind: {artifact.kind}",
            http_status=409,
        )
