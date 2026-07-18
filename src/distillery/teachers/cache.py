"""Write-once, content-verified teacher response cache."""

from __future__ import annotations

from distillery.contracts.hashing import content_sha256
from distillery.teachers.errors import TeacherErrorCode, raise_teacher_error
from distillery.teachers.types import TeacherResult


class ImmutableTeacherCache:
    """In-memory contract suitable for replacement with immutable storage."""

    def __init__(self) -> None:
        self._entries: dict[str, TeacherResult] = {}
        self._entry_digests: dict[str, str] = {}

    def get(self, key: str) -> TeacherResult | None:
        result = self._entries.get(key)
        if result is None:
            return None
        digest = content_sha256(result.canonical_body())
        if digest != result.result_sha256 or digest != self._entry_digests.get(key):
            raise_teacher_error(
                TeacherErrorCode.CACHE_INTEGRITY_FAILED,
                "cached teacher result failed content verification",
                details={"cache_key": key, "actual_sha256": digest},
            )
        return result

    def put(self, key: str, result: TeacherResult) -> TeacherResult:
        digest = content_sha256(result.canonical_body())
        if digest != result.result_sha256:
            raise_teacher_error(
                TeacherErrorCode.CACHE_INTEGRITY_FAILED,
                "teacher result hash is invalid",
                details={
                    "cache_key": key,
                    "expected_sha256": digest,
                    "actual_sha256": result.result_sha256,
                },
            )
        existing = self._entries.get(key)
        if existing is not None:
            existing_digest = self._entry_digests[key]
            if existing_digest != digest:
                raise_teacher_error(
                    TeacherErrorCode.CACHE_INTEGRITY_FAILED,
                    "immutable teacher cache key already has different content",
                    details={
                        "cache_key": key,
                        "existing_sha256": existing_digest,
                        "incoming_sha256": digest,
                    },
                )
            return existing
        self._entries[key] = result
        self._entry_digests[key] = digest
        return result

    def __len__(self) -> int:
        return len(self._entries)


__all__ = ["ImmutableTeacherCache"]
