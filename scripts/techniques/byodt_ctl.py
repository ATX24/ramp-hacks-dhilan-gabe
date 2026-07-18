#!/usr/bin/env python3
"""BYODT CLI: validate / register / plan-only (never launches training)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from distillery.techniques import (
    CompatibilityContext,
    TechniqueDescriptor,
    TechniqueError,
    TechniqueRegistry,
    TechniqueRequest,
)
from distillery.techniques.channel import load_channel_plan, write_channel_plan


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return payload


def cmd_validate_descriptor(args: argparse.Namespace) -> int:
    payload = _load_json(Path(args.descriptor))
    if "descriptor_sha256" in payload:
        descriptor = TechniqueDescriptor.model_validate(payload)
    else:
        descriptor = TechniqueDescriptor.seal(**payload)
    descriptor.assert_integrity()
    print(
        json.dumps(
            {
                "ok": True,
                "technique_id": descriptor.technique_id,
                "version": descriptor.version,
                "descriptor_sha256": descriptor.descriptor_sha256,
                "execution": descriptor.execution.value,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    registry_dir = Path(args.registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry = TechniqueRegistry.with_builtins()
    # Reload previously registered externals for collision checks.
    for path in sorted(registry_dir.glob("*.json")):
        registry.register_from_path(path)
    descriptor = registry.register_from_path(Path(args.descriptor))
    out = registry_dir / f"{descriptor.technique_id}@{descriptor.version}.json".replace("/", "_")
    out.write_text(
        json.dumps(descriptor.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "registered": descriptor.technique_key,
                "path": str(out),
                "descriptor_sha256": descriptor.descriptor_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    registry = TechniqueRegistry.with_builtins()
    if args.descriptor:
        registry.register_from_path(Path(args.descriptor))
    if args.registry_dir:
        for path in sorted(Path(args.registry_dir).glob("*.json")):
            registry.register_from_path(path)
    request = TechniqueRequest(
        technique_id=args.technique_id,
        version=args.version,
        config=_load_json(Path(args.config)),
    )
    context = CompatibilityContext.model_validate(_load_json(Path(args.context)))
    plan = registry.plan(request, context)
    payload = plan.model_dump(mode="json")
    if args.channel_dir:
        channel_dir = Path(args.channel_dir)
        if plan.external_execution is None:
            raise SystemExit("plan has no external execution channel")
        write_channel_plan(channel_dir, plan=plan)
        envelope = load_channel_plan(channel_dir)
        payload["channel_written"] = str(channel_dir)
        payload["channel_hash"] = envelope.envelope_sha256
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="byodt_ctl",
        description="Validate, register, or plan Distillery techniques (no training).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser(
        "validate-descriptor",
        help="Validate descriptor/schema only (use plan for compatibility preflight)",
    )
    validate.add_argument("descriptor", help="Path to technique JSON")
    validate.set_defaults(func=cmd_validate_descriptor)

    register = sub.add_parser("register", help="Register a sealed external descriptor")
    register.add_argument("descriptor", help="Path to technique JSON")
    register.add_argument(
        "--registry-dir",
        required=True,
        help="Directory of registered technique descriptors",
    )
    register.set_defaults(func=cmd_register)

    plan = sub.add_parser("plan", help="Plan-only technique resolution")
    plan.add_argument("--technique-id", required=True)
    plan.add_argument("--version", required=True)
    plan.add_argument("--config", required=True, help="Config JSON path")
    plan.add_argument("--context", required=True, help="Compatibility context JSON")
    plan.add_argument("--descriptor", help="Optional external descriptor to load")
    plan.add_argument("--registry-dir", help="Optional registry directory")
    plan.add_argument(
        "--channel-dir",
        help="Optional directory to materialize technique_plan.json",
    )
    plan.set_defaults(func=cmd_plan)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except TechniqueError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "code": exc.code.value,
                    "message": exc.payload.message,
                    "details": dict(exc.payload.details),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
