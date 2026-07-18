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
    """
    Configure FA2 only when compatibility is attested.

    Callers inject probe results in tests; production may supply live probes.
    """
    if not requested:
        return FlashAttentionAttestation(
            requested=False,
            attested=False,
            torch_version=torch_version or "unknown",
            cuda_available=bool(cuda_available),
            flash_attn_importable=bool(flash_attn_importable),
            reason="flash_attention_2_not_requested",
        )

    tv = torch_version
    cuda = cuda_available
    fa = flash_attn_importable
    if tv is None or cuda is None or fa is None:
        raise FlashAttentionError(
            "FlashAttention 2 requested but compatibility probes were not supplied"
        )
    if not cuda:
        raise FlashAttentionError("FlashAttention 2 requested but CUDA is unavailable")
    if not fa:
        raise FlashAttentionError("FlashAttention 2 requested but flash_attn is not importable")
    # Torch 2.4.x is the sealed training-image line.
    if not tv.startswith("2.4."):
        raise FlashAttentionError(
            f"FlashAttention 2 not attested for torch_version={tv!r}; require 2.4.x"
        )
    return FlashAttentionAttestation(
        requested=True,
        attested=True,
        torch_version=tv,
        cuda_available=True,
        flash_attn_importable=True,
        reason="torch_2.4_x_cuda_flash_attn_importable",
    )


def attn_implementation_for(attestation: FlashAttentionAttestation) -> str:
    if attestation.requested and not attestation.attested:
        raise FlashAttentionError("cannot enable FA2 without attestation")
    return "flash_attention_2" if attestation.attested else "sdpa"
