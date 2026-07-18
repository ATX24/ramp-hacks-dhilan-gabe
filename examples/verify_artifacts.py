"""Verify Distillery artifact trees against a SHA256SUMS manifest.

Compatible with the planned S3 layout:

  integrity/SHA256SUMS
  model/adapter/*
  evaluation/predictions.jsonl
  ...

Fails loud on missing files, unexpected extras (optional), or digest mismatch.
Never downloads models or starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    checked: int
    mismatches: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    sums_path: str

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "checked": self.checked,
            "mismatches": list(self.mismatches),
            "missing": list(self.missing),
            "unexpected": list(self.unexpected),
            "sums_path": self.sums_path,
            "claim": (
                "Integrity check of local/precomputed artifacts only. "
                "Does not assert scientific proof status; real benchmarks pending."
            ),
        }


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse GNU ``sha256sum`` / ``SHA256SUMS`` lines: ``<hex>  <path>``."""
    mapping: dict[str, str] = {}
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"SHA256SUMS:{line_no}: expected '<hex>  <path>'")
        digest, rel = parts
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError(f"SHA256SUMS:{line_no}: invalid sha256 digest")
        rel = rel[2:] if rel.startswith("*") else rel
        rel = rel.lstrip("./")
        mapping[rel] = digest.lower()
    if not mapping:
        raise ValueError("SHA256SUMS is empty")
    return mapping


def load_expected_from_fixture_manifest(path: Path) -> dict[str, str]:
    """Optional helper for the golden fixture_manifest.json shape."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"{path}: missing files map")
    out: dict[str, str] = {}
    for name, meta in files.items():
        if not isinstance(meta, Mapping) or "sha256" not in meta:
            raise ValueError(f"{path}: files[{name!r}] missing sha256")
        out[str(name)] = str(meta["sha256"]).lower()
    return out


def verify_tree(
    root: Path,
    expected: Mapping[str, str],
    *,
    allow_unexpected: bool = True,
    sums_path: str = "",
) -> VerifyResult:
    mismatches: list[str] = []
    missing: list[str] = []
    checked = 0

    for rel, want in sorted(expected.items()):
        path = root / rel
        if not path.is_file():
            missing.append(rel)
            continue
        got = sha256_file(path)
        checked += 1
        if got != want.lower():
            mismatches.append(f"{rel}: expected={want.lower()} got={got}")

    unexpected: list[str] = []
    if not allow_unexpected:
        expected_names = set(expected)
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = path.relative_to(root).as_posix()
            if rel in {"SHA256SUMS", "integrity/SHA256SUMS"}:
                continue
            if rel not in expected_names:
                unexpected.append(rel)

    ok = not mismatches and not missing and not unexpected
    return VerifyResult(
        ok=ok,
        checked=checked,
        mismatches=tuple(mismatches),
        missing=tuple(missing),
        unexpected=tuple(unexpected),
        sums_path=sums_path,
    )


def verify_from_sha256sums(
    root: Path,
    sums_file: Path | None = None,
    *,
    allow_unexpected: bool = True,
) -> VerifyResult:
    candidates: Iterable[Path]
    if sums_file is not None:
        candidates = (sums_file,)
    else:
        candidates = (
            root / "integrity" / "SHA256SUMS",
            root / "SHA256SUMS",
        )
    sums_path: Path | None = None
    for candidate in candidates:
        if candidate.is_file():
            sums_path = candidate
            break
    if sums_path is None:
        return VerifyResult(
            ok=False,
            checked=0,
            mismatches=(),
            missing=("SHA256SUMS",),
            unexpected=(),
            sums_path="",
        )
    expected = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    return verify_tree(
        root,
        expected,
        allow_unexpected=allow_unexpected,
        sums_path=str(sums_path),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Artifact directory root (local copy of a run or proof package)",
    )
    parser.add_argument(
        "--sums",
        type=Path,
        default=None,
        help="Explicit SHA256SUMS path (default: root/integrity/SHA256SUMS)",
    )
    parser.add_argument(
        "--fixture-manifest",
        type=Path,
        default=None,
        help="Alternate: verify against tests/fixtures/.../fixture_manifest.json files map",
    )
    parser.add_argument(
        "--strict-tree",
        action="store_true",
        help="Fail if unexpected files exist under root",
    )
    args = parser.parse_args(argv)

    root = args.root
    if not root.is_dir():
        print(json.dumps({"ok": False, "error": f"not a directory: {root}"}), file=sys.stderr)
        return 2

    if args.fixture_manifest is not None:
        expected = load_expected_from_fixture_manifest(args.fixture_manifest)
        result = verify_tree(
            root,
            expected,
            allow_unexpected=not args.strict_tree,
            sums_path=str(args.fixture_manifest),
        )
    else:
        result = verify_from_sha256sums(
            root,
            args.sums,
            allow_unexpected=not args.strict_tree,
        )

    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
