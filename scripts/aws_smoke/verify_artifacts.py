#!/usr/bin/env python3
"""Verify emergency run artifact checksums (local filesystem only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from experiments.aws_smoke.artifacts import verify_emergency_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(prog="verify_artifacts")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = verify_emergency_artifacts(args.root)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
