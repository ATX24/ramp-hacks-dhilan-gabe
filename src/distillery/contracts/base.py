"""Deeply immutable building blocks for public contract models."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import (
    Annotated,
    Any,
    Self,
    TypeVar,
)

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    JsonValue,
    PlainSerializer,
    StrictStr,
)

from distillery.contracts.hashing import RFC8785_SAFE_INTEGER_MAX

K = TypeVar("K")
V = TypeVar("V")


def _deep_freeze(value: Any, *, path: str = "$") -> Any:
    if isinstance(value, BaseModel):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item, path=f"{path}.{key}") for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _deep_freeze(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        )
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > RFC8785_SAFE_INTEGER_MAX:
            raise ValueError(f"{path}: integer exceeds RFC 8785 interoperable safe domain")
    return value


def deep_thaw(value: Any) -> Any:
    """Return ordinary JSON containers for serialization and validated copying."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return {key: deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [deep_thaw(item) for item in value]
    return value


def _freeze_mapping(value: dict[K, V]) -> Mapping[K, V]:
    return _deep_freeze(value)


def _freeze_json_object(
    value: dict[StrictStr, JsonValue],
) -> Mapping[StrictStr, JsonValue]:
    return _deep_freeze(value)


def _normalize_json_containers(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _normalize_json_containers(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {key: _normalize_json_containers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_containers(item) for item in value]
    return value


FrozenDict = Annotated[
    dict[K, V],
    AfterValidator(_freeze_mapping),
    PlainSerializer(deep_thaw, return_type=dict),
]
FrozenJsonObject = Annotated[
    dict[StrictStr, JsonValue],
    BeforeValidator(_normalize_json_containers),
    AfterValidator(_freeze_json_object),
    PlainSerializer(deep_thaw, return_type=dict),
]


class FrozenModel(BaseModel):
    """Frozen Pydantic model whose validated copies cannot bypass invariants."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
    )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        del deep
        payload = self.model_dump(mode="python", round_trip=True)
        if update is not None:
            payload.update(dict(update))
        return type(self).model_validate(payload)
