"""Static readiness checks for the emergency trainer (no weight download)."""

from __future__ import annotations

import ast
from pathlib import Path

TRAIN_MODULE = Path(__file__).with_name("train.py")

REQUIRED_TOP_LEVEL_IMPORTS: frozenset[str] = frozenset(
    {
        "torch",
        "peft",
        "transformers",
    }
)

REQUIRED_SYMBOLS: frozenset[str] = frozenset(
    {
        "run_training",
        "compute_torch_loss",
        "load_student",
        "load_teacher",
        "forward_kl_chunked_torch",
        "hard_cross_entropy_torch",
    }
)


def parse_train_module(path: Path | None = None) -> ast.Module:
    target = path or TRAIN_MODULE
    return ast.parse(target.read_text(encoding="utf-8"), filename=str(target))


def top_level_imported_modules(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def defined_function_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def assert_trainer_source_ready(path: Path | None = None) -> dict[str, object]:
    """
    Fail loud if the emergency trainer source lacks real ML imports/symbols.

    This does not import torch at runtime; it proves the source wiring exists.
    """
    tree = parse_train_module(path)
    modules = top_level_imported_modules(tree)
    missing_modules = sorted(REQUIRED_TOP_LEVEL_IMPORTS - modules)
    if missing_modules:
        raise RuntimeError(
            "train.py missing required top-level ML imports: "
            + ", ".join(missing_modules)
        )
    defined = defined_function_names(tree)
    imported = imported_names(tree)
    available = defined | imported
    missing_symbols = sorted(REQUIRED_SYMBOLS - available)
    if missing_symbols:
        raise RuntimeError(
            "train.py missing required trainer symbols: " + ", ".join(missing_symbols)
        )
    return {
        "ok": True,
        "modules": sorted(modules),
        "symbols": sorted(REQUIRED_SYMBOLS),
    }
