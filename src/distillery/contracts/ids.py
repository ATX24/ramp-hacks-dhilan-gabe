"""Immutable, prefix-validated resource identifiers."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, StrictStr

_ID_BODY = re.compile(r"^[a-z0-9][a-z0-9_-]{1,126}$")


def _validate_prefixed(prefix: str, value: str) -> str:
    if not value.startswith(prefix):
        raise ValueError(f"id must start with {prefix!r}, got {value!r}")
    body = value[len(prefix) :]
    if not _ID_BODY.match(body):
        raise ValueError(
            f"id body after {prefix!r} must match {_ID_BODY.pattern}, got {body!r}"
        )
    return value


def _dataset_id(v: str) -> str:
    return _validate_prefixed("ds_", v)


def _run_id(v: str) -> str:
    return _validate_prefixed("run_", v)


def _artifact_id(v: str) -> str:
    return _validate_prefixed("art_", v)


def _proof_id(v: str) -> str:
    return _validate_prefixed("prf_", v)


def _example_id(v: str) -> str:
    return _validate_prefixed("ex_", v)


def _world_id(v: str) -> str:
    return _validate_prefixed("world_", v)


def _group_id(v: str) -> str:
    return _validate_prefixed("grp_", v)


DatasetId = Annotated[StrictStr, AfterValidator(_dataset_id), Field(min_length=4)]
RunId = Annotated[StrictStr, AfterValidator(_run_id), Field(min_length=5)]
ArtifactId = Annotated[StrictStr, AfterValidator(_artifact_id), Field(min_length=5)]
ProofReportId = Annotated[StrictStr, AfterValidator(_proof_id), Field(min_length=5)]
ExampleId = Annotated[StrictStr, AfterValidator(_example_id), Field(min_length=4)]
WorldId = Annotated[StrictStr, AfterValidator(_world_id), Field(min_length=7)]
GroupId = Annotated[StrictStr, AfterValidator(_group_id), Field(min_length=5)]

ResourceKind = Literal["dataset", "run", "artifact", "proof_report"]
ResourceId = DatasetId | RunId | ArtifactId | ProofReportId
