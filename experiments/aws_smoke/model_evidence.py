"""Offline model-snapshot and tokenizer evidence checks.

The emergency trainer never resolves a Hugging Face model id at runtime. Each
model must exist at ``models/<org>/<name>/<40-hex-revision>/`` and tokenizer
evidence is computed from regular files in that exact directory.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from distillery.contracts.hashing import content_sha256
from experiments.aws_smoke.loss_wiring import assert_special_token_maps_compatible

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_TOKENIZER_FILENAMES = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
)
_TOKENIZER_REQUIRED_ANY = (
    frozenset({"tokenizer.json"}),
    frozenset({"tokenizer.model"}),
    frozenset({"sentencepiece.bpe.model"}),
    frozenset({"vocab.json", "merges.txt"}),
)


class LoadedTokenizer(Protocol):
    chat_template: str | None
    special_tokens_map: dict[str, Any]

    def convert_tokens_to_ids(self, tokens: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class TokenizerRuntimeEvidence:
    snapshot_dir: Path
    tokenizer_sha256: str
    chat_template_sha256: str
    special_token_map: dict[str, int]
    file_sha256: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_dir": str(self.snapshot_dir),
            "tokenizer_sha256": self.tokenizer_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "special_token_map": self.special_token_map,
            "file_sha256": self.file_sha256,
        }


def require_local_snapshot(
    models_dir: Path,
    model_id: str,
    revision: str,
) -> Path:
    """Return the exact revision directory or fail before any model API call."""
    if _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("model revision must be exactly 40 lowercase hex characters")
    org, separator, name = model_id.partition("/")
    if not separator or not org or not name or "/" in name:
        raise ValueError(f"model id must have exact org/name form, got {model_id!r}")
    candidate = models_dir / org / name / revision
    if not candidate.is_dir() or candidate.is_symlink():
        raise FileNotFoundError(
            "missing exact offline model snapshot directory: "
            f"{candidate}; network fallback is forbidden"
        )
    config = candidate / "config.json"
    if not _is_regular_nonempty(config):
        raise FileNotFoundError(f"offline snapshot lacks nonempty config.json: {candidate}")
    return candidate


def tokenizer_snapshot_file_hashes(snapshot_dir: Path) -> dict[str, str]:
    """Hash every recognized tokenizer file in one exact revision directory."""
    entries: dict[str, str] = {}
    for path in sorted(snapshot_dir.iterdir()):
        if path.name not in _TOKENIZER_FILENAMES:
            continue
        if not _is_regular_nonempty(path):
            raise ValueError(f"tokenizer evidence file must be regular and nonempty: {path}")
        entries[path.name] = _sha256_path(path)
    if "tokenizer_config.json" not in entries:
        raise FileNotFoundError(f"missing tokenizer_config.json in {snapshot_dir}")
    names = frozenset(entries)
    if not any(required <= names for required in _TOKENIZER_REQUIRED_ANY):
        raise FileNotFoundError(
            "snapshot lacks tokenizer.json, tokenizer model, or vocab+merges: "
            f"{snapshot_dir}"
        )
    return entries


def verify_model_config_sha256(snapshot_dir: Path, expected_sha256: str) -> str:
    config_path = snapshot_dir / "config.json"
    if not _is_regular_nonempty(config_path):
        raise FileNotFoundError(f"missing nonempty model config: {config_path}")
    actual = _sha256_path(config_path)
    if actual != expected_sha256:
        raise ValueError(
            "local model config hash differs from sealed evidence: "
            f"expected={expected_sha256} actual={actual} path={config_path}"
        )
    return actual


def require_local_model_weights(snapshot_dir: Path) -> tuple[Path, ...]:
    """Require nonempty regular weight files before calling Transformers."""
    candidates = sorted(snapshot_dir.glob("*.safetensors")) + sorted(
        snapshot_dir.glob("pytorch_model*.bin")
    )
    weights = tuple(path for path in candidates if _is_regular_nonempty(path))
    if not weights:
        raise FileNotFoundError(
            f"offline snapshot lacks nonempty local model weights: {snapshot_dir}"
        )
    invalid = [path for path in candidates if not _is_regular_nonempty(path)]
    if invalid:
        raise ValueError(f"model weight files must be regular and nonempty: {invalid}")
    return weights


def tokenizer_snapshot_sha256(snapshot_dir: Path) -> tuple[str, dict[str, str]]:
    entries = tokenizer_snapshot_file_hashes(snapshot_dir)
    return content_sha256({"tokenizer_files": entries}), entries


def chat_template_sha256(chat_template: str | None) -> str:
    if chat_template is None or not chat_template.strip():
        raise ValueError("loaded tokenizer has no nonempty pinned chat template")
    return hashlib.sha256(chat_template.encode("utf-8")).hexdigest()


def loaded_special_token_map(
    tokenizer: LoadedTokenizer,
) -> dict[str, int]:
    """Return a stable map from special-token roles to loaded token ids."""
    result: dict[str, int] = {}
    for role, token_value in sorted(tokenizer.special_tokens_map.items()):
        if isinstance(token_value, (list, tuple)):
            ids = tokenizer.convert_tokens_to_ids([str(value) for value in token_value])
            if not isinstance(ids, list):
                ids = list(ids)
            for index, value in enumerate(ids):
                result[f"{role}[{index}]"] = int(value)
        else:
            token_id = tokenizer.convert_tokens_to_ids(str(token_value))
            if isinstance(token_id, list):
                if len(token_id) != 1:
                    raise ValueError(f"special token role {role!r} resolved to multiple ids")
                token_id = token_id[0]
            result[str(role)] = int(token_id)
    if not result:
        raise ValueError("loaded tokenizer special-token map is empty")
    return result


def collect_tokenizer_runtime_evidence(
    snapshot_dir: Path,
    tokenizer: LoadedTokenizer,
) -> TokenizerRuntimeEvidence:
    tokenizer_digest, file_hashes = tokenizer_snapshot_sha256(snapshot_dir)
    return TokenizerRuntimeEvidence(
        snapshot_dir=snapshot_dir,
        tokenizer_sha256=tokenizer_digest,
        chat_template_sha256=chat_template_sha256(tokenizer.chat_template),
        special_token_map=loaded_special_token_map(tokenizer),
        file_sha256=file_hashes,
    )


def verify_tokenizer_runtime_evidence(
    *,
    snapshot_dir: Path,
    tokenizer: LoadedTokenizer,
    expected_tokenizer_sha256: str,
    expected_chat_template_sha256: str,
    expected_special_token_map: dict[str, int],
) -> TokenizerRuntimeEvidence:
    actual = collect_tokenizer_runtime_evidence(snapshot_dir, tokenizer)
    if actual.tokenizer_sha256 != expected_tokenizer_sha256:
        raise ValueError(
            "loaded tokenizer files do not match sealed evidence: "
            f"expected={expected_tokenizer_sha256} actual={actual.tokenizer_sha256}"
        )
    if actual.chat_template_sha256 != expected_chat_template_sha256:
        raise ValueError(
            "loaded chat template does not match sealed evidence: "
            f"expected={expected_chat_template_sha256} "
            f"actual={actual.chat_template_sha256}"
        )
    if actual.special_token_map != expected_special_token_map:
        raise ValueError(
            "loaded special-token map does not match sealed evidence: "
            f"expected={expected_special_token_map} actual={actual.special_token_map}"
        )
    return actual


def assert_loaded_tokenizers_compatible(
    teacher: TokenizerRuntimeEvidence,
    student: TokenizerRuntimeEvidence,
) -> None:
    if teacher.tokenizer_sha256 != student.tokenizer_sha256:
        raise ValueError("teacher/student loaded tokenizer file hashes differ")
    if teacher.chat_template_sha256 != student.chat_template_sha256:
        raise ValueError("teacher/student loaded chat template hashes differ")
    assert_special_token_maps_compatible(
        teacher.special_token_map,
        student.special_token_map,
    )


def write_tokenizer_evidence(
    path: Path,
    *,
    student: TokenizerRuntimeEvidence,
    teacher: TokenizerRuntimeEvidence,
) -> None:
    payload = {
        "schema_version": "distillery.aws_smoke.loaded_tokenizers.v1",
        "student": student.to_dict(),
        "teacher": teacher.to_dict(),
        "compatible": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_regular_nonempty(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    return not path.is_symlink() and stat.S_ISREG(mode) and path.stat().st_size > 0


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
