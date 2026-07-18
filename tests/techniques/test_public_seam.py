"""The package exposes one small planning seam, not its implementation."""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

import distillery.techniques as techniques
from distillery.techniques.adapters.external import ExternalContainerAdapter


def test_public_seam_is_narrow() -> None:
    assert set(techniques.__all__) == {
        "CompatibilityContext",
        "TechniqueDescriptor",
        "TechniqueError",
        "TechniqueErrorCode",
        "TechniquePlan",
        "TechniqueRegistry",
        "TechniqueRequest",
        "recompute_protocol_hash",
    }
    for internal in (
        "ArtifactContract",
        "ExternalContainerAdapter",
        "TechniqueChannelContract",
        "advance_lifecycle",
        "load_channel_plan",
        "negotiate_compatibility",
        "write_channel_plan",
    ):
        assert not hasattr(techniques, internal)


def test_external_descriptor_has_no_dynamic_import_entrypoint(
    external_descriptor,
) -> None:
    payload = external_descriptor.canonical_payload()
    payload["python_module"] = "untrusted.plugin"
    with pytest.raises(ValidationError):
        techniques.TechniqueDescriptor.seal(**payload)
    source = inspect.getsource(ExternalContainerAdapter)
    assert "importlib" not in source
    assert "import_module" not in source
