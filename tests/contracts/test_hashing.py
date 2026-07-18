"""RFC 8785 canonicalization and fail-loud hashing behavior."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import BaseModel

from distillery.contracts.hashing import (
    RFC8785_SAFE_INTEGER_MAX,
    canonical_json_bytes,
    content_sha256,
    sha256_hex,
)


class HashableModel(BaseModel):
    at: datetime
    label: str


def test_rfc8785_canonicalization_is_cross_language_stable() -> None:
    value = {
        "string": '€$\u000f\nA\'B"\\\\"/',
        "numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 1e-27],
        "literal": [None, True, False],
    }
    assert canonical_json_bytes(value) == (
        b'{"literal":[null,true,false],'
        b'"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'
        b'"string":"\xe2\x82\xac$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}'
    )


def test_explicit_pydantic_and_date_normalization() -> None:
    instant = datetime(2026, 7, 18, 12, 0, tzinfo=timezone(timedelta(hours=-4)))
    model = HashableModel(at=instant, label="demo")
    assert canonical_json_bytes(model) == (b'{"at":"2026-07-18T16:00:00.000000Z","label":"demo"}')
    assert canonical_json_bytes(date(2026, 7, 18)) == b'"2026-07-18"'
    assert content_sha256(model) == content_sha256(
        {
            "at": datetime(2026, 7, 18, 16, 0, tzinfo=UTC),
            "label": "demo",
        }
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_rejected_at_any_depth(bad: float) -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"nested": [bad]})


@pytest.mark.parametrize(
    "bad",
    [
        {"value": Decimal("1.0")},
        {"value": {1, 2}},
        {"value": b"bytes"},
        {1: "non-string key"},
        {"value": object()},
    ],
)
def test_unsupported_non_json_values_rejected(bad: object) -> None:
    with pytest.raises(TypeError):
        canonical_json_bytes(bad)


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError, match="naive datetime"):
        canonical_json_bytes(datetime(2026, 7, 18, 12, 0))


def test_tuples_normalize_explicitly_to_json_arrays() -> None:
    assert canonical_json_bytes(("a", 1, True)) == b'["a",1,true]'


def test_sha256_hex_requires_bytes() -> None:
    with pytest.raises(TypeError, match="requires bytes"):
        sha256_hex(bytearray(b"not-bytes"))  # type: ignore[arg-type]


def test_non_interoperable_integer_rejected_before_rfc8785_encoder() -> None:
    with pytest.raises(ValueError, match="safe domain"):
        canonical_json_bytes({"value": RFC8785_SAFE_INTEGER_MAX + 1})
