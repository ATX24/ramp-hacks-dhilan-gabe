"""FlashAttention 2 compatibility attestation gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class FlashAttentionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FlashAttentionAttestation:
    requested: bool
    attested: bool
    torch_version: str
    cuda_available: bool
    flash_attn_importable: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "attested": self.attested,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "flash_attn_importable": self.flash_attn_importable,
            "reason": self.reason,
        }


def attest_flash_attention_2(
    *,
    requested: bool,
    torch_version: str | None = None,
    cuda_available: bool | None = None,
    flash_attn_importable: bool | None = None,
) -> FlashAttentionAttestation:
    if not requested:
        return FlashAttentionAttestation(
            requested=False,
            attested=False,
            torch_version=torch_version or "unknown",
            cuda_available=bool(cuda_available),
            flash_attn_importable=bool(flash_attn_importable),
            reason="flash_attention_2_not_requested",
        )
    if torch_version is None or cuda_available is None or flash_attn_importable is None:
        raise FlashAttentionError(
            "FlashAttention 2 requested but compatibility probes were not supplied"
        )
    if not cuda_available:
        raise FlashAttentionError("FlashAttention 2 requested but CUDA is unavailable")
    if not flash_attn_importable:
        raise FlashAttentionError(
            "FlashAttention 2 requested but flash_attn is not importable"
        )
    if not torch_version.startswith("2.4."):
        raise FlashAttentionError(
            f"FlashAttention 2 not attested for torch_version={torch_version!r}; require 2.4.x"
        )
    return FlashAttentionAttestation(
        requested=True,
        attested=True,
        torch_version=torch_version,
        cuda_available=True,
        flash_attn_importable=True,
        reason="torch_2.4_x_cuda_flash_attn_importable",
    )


def attn_implementation_for(attestation: FlashAttentionAttestation) -> str:
    if attestation.requested and not attestation.attested:
        raise FlashAttentionError("cannot enable FA2 without attestation")
    return "flash_attention_2" if attestation.attested else "sdpa"
