"""Ensure base package imports without ML extras; torch module stays optional."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_recipes_and_training_import_without_torch() -> None:
    # Drop optional torch module if a prior test imported it.
    sys.modules.pop("distillery.training.torch_losses", None)
    recipes = importlib.import_module("distillery.recipes")
    training = importlib.import_module("distillery.training")
    assert hasattr(recipes, "SequenceV1Recipe")
    assert hasattr(training, "forward_kl_chunked")
    assert hasattr(training, "run_entrypoint")
    # Optional module must not be pulled in by package __init__.
    assert "distillery.training.torch_losses" not in sys.modules


def test_optional_torch_ce_declares_ignore_index_without_importing_torch() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "distillery"
        / "training"
        / "torch_losses.py"
    ).read_text(encoding="utf-8")
    assert "ignore_index: int = -100" in source
    assert "ignore_index=ignore_index" in source
    assert "target/mask alignment" in source
    assert "import torch" in source
    assert "distillery.training.torch_losses" not in sys.modules
