"""Model-facing inputs contain context, never latent labels or answer helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from distillery.contracts.tasks import TaskId
from distillery.data.validate import (
    FORBIDDEN_INPUT_KEYS,
    find_input_hygiene_errors,
)


def test_full_corpus_input_denylist_and_target_copy_gate(full_corpus) -> None:
    for example in full_corpus.examples:
        assert (
            find_input_hygiene_errors(
                example.input,
                expected_output=example.expected_output,
            )
            == ()
        )
        keys = {path[-1].casefold() for path, _value in _walk(example.input) if path}
        assert keys.isdisjoint(FORBIDDEN_INPUT_KEYS)
        assert "gl_candidates" not in keys
        assert "case_nonce" not in keys
        assert "regime" not in keys


def test_variance_targets_have_no_deterministic_copy_field(full_corpus) -> None:
    """MI-style proxy: no fixed numeric input path copies the exact target."""
    matches_by_path: dict[str, int] = defaultdict(int)
    observations_by_path: dict[str, int] = defaultdict(int)
    variances = [
        example for example in full_corpus.examples if example.task == TaskId.VARIANCE_ANALYSIS
    ]
    for example in variances:
        target = example.expected_output["profit_impact_minor"]
        for path, value in _walk(example.input):
            if type(value) is not int:
                continue
            generalized = ".".join("*" if component.isdigit() else component for component in path)
            observations_by_path[generalized] += 1
            if value == target:
                matches_by_path[generalized] += 1
    assert variances
    for path, observations in observations_by_path.items():
        match_rate = matches_by_path[path] / observations
        assert match_rate < 0.95, (path, match_rate)


def test_variance_driver_impacts_are_derived_not_copied(full_corpus) -> None:
    for example in full_corpus.examples:
        if example.task != TaskId.VARIANCE_ANALYSIS:
            continue
        numeric_leaves = {
            value
            for path, value in _walk(example.input)
            if type(value) is int
            and path
            and any(
                marker in path[-1].casefold() for marker in ("hint", "impact", "target", "answer")
            )
        }
        expected_impacts = {
            driver["impact_minor"] for driver in example.expected_output["top_drivers"]
        }
        assert numeric_leaves.isdisjoint(expected_impacts)


def _walk(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _walk(value[key], (*path, str(key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            yield from _walk(item, (*path, str(index)))
    else:
        yield path, value
