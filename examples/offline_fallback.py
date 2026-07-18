"""Offline / precomputed demo fallback.

Loads a checksum-verified local artifact package and a static proof report so
the two-minute pitch can proceed without network, AWS, or live training.

Honest labeling: this mode never invents metrics. It only presents artifacts
that pass integrity checks, and always surfaces that real live benchmarks may
still be pending.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verify_artifacts import sha256_file, verify_from_sha256sums


@dataclass(frozen=True)
class OfflineDemoBundle:
    artifact_root: Path
    report_path: Path
    proof_status: str
    limitations: tuple[str, ...]
    verified: bool
    verify_detail: dict[str, Any]

    def stage_payload(self) -> dict[str, Any]:
        return {
            "mode": "precomputed_offline",
            "verified": self.verified,
            "artifact_root": str(self.artifact_root),
            "report_path": str(self.report_path),
            "proof_status": self.proof_status,
            "limitations": list(self.limitations),
            "verify": self.verify_detail,
            "ui_label": "PRECOMPUTED ARTIFACTS — checksum verified; not a live run",
            "claim": (
                "Offline fallback presents immutable precomputed artifacts only. "
                "Projected serving economics remain projected. "
                "Real benchmark completion may still be pending."
            ),
        }


def _default_limitations(report: dict[str, Any]) -> tuple[str, ...]:
    raw = report.get("limitations")
    base: list[str] = []
    if isinstance(raw, list):
        base.extend(str(item) for item in raw)
    base.extend(
        [
            "Demo data is synthetic (finance_world.v1); not customer traces.",
            "Serving cost figures, if present, are projected unless labeled measured.",
            "Offline mode does not re-run training or SageMaker jobs.",
        ]
    )
    # Deduplicate preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for item in base:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def load_offline_bundle(
    artifact_root: Path,
    report_path: Path,
    *,
    require_verify: bool = True,
) -> OfflineDemoBundle:
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"artifact root missing: {artifact_root}")
    if not report_path.is_file():
        raise FileNotFoundError(f"proof report missing: {report_path}")

    verify = verify_from_sha256sums(artifact_root)
    verify_detail = verify.as_dict()
    if require_verify and not verify.ok:
        raise RuntimeError(
            "artifact integrity failed; refuse offline demo. "
            f"detail={json.dumps(verify_detail, sort_keys=True)}"
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("proof report must be a JSON object")
    status = str(report.get("proof_status", "insufficient_evidence"))
    return OfflineDemoBundle(
        artifact_root=artifact_root,
        report_path=report_path,
        proof_status=status,
        limitations=_default_limitations(report),
        verified=verify.ok,
        verify_detail=verify_detail,
    )


def write_stub_precomputed_layout(target: Path) -> Path:
    """Create a tiny local layout for rehearsals when real run artifacts are absent.

    Stub metrics are explicitly labeled ``stub_not_a_benchmark`` so they cannot
    be mistaken for measured results.
    """
    target.mkdir(parents=True, exist_ok=True)
    integrity = target / "integrity"
    integrity.mkdir(exist_ok=True)
    report = {
        "schema_version": "distillery.proof_report.v1",
        "report_id": "prf_stub_offline_001",
        "run_ids": ["run_stub_offline_001"],
        "protocol_id": "finance-proof.v1",
        "protocol_sha256": "0" * 64,
        "proof_status": "insufficient_evidence",
        "first_failed_gate": "evidence",
        "unevaluated_gates": ["quality", "economics"],
        "arm_results": [],
        "quality_gates": [],
        "uncertainty": {},
        "economics": {"label": "stub_not_a_benchmark"},
        "exclusions": ["all_arms_pending"],
        "limitations": [
            "Stub offline package for rehearsal only; real benchmarks pending.",
        ],
        "created_at": "2026-07-18T00:00:00+00:00",
    }
    report_path = target / "report.json"
    report_bytes = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report_path.write_bytes(report_bytes)

    predictions = target / "evaluation" / "predictions.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    pred_line = json.dumps(
        {"example_id": "ex_stub_001", "arm_id": "rules", "label": "stub_not_a_benchmark"},
        sort_keys=True,
    )
    predictions.write_text(pred_line + "\n", encoding="utf-8")

    lines = [
        f"{sha256_file(report_path)}  report.json",
        f"{sha256_file(predictions)}  evaluation/predictions.jsonl",
    ]
    (integrity / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to report.json (default: <artifact-root>/report.json)",
    )
    parser.add_argument(
        "--write-stub",
        action="store_true",
        help="Create a clearly labeled stub package under --artifact-root",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Dangerous: proceed even if checksums fail (prints verified=false)",
    )
    args = parser.parse_args(argv)

    if args.write_stub:
        report_path = write_stub_precomputed_layout(args.artifact_root)
        print(
            json.dumps(
                {
                    "wrote_stub": True,
                    "artifact_root": str(args.artifact_root),
                    "report": str(report_path),
                    "warning": "stub_not_a_benchmark",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    report_path = args.report or (args.artifact_root / "report.json")
    try:
        bundle = load_offline_bundle(
            args.artifact_root,
            report_path,
            require_verify=not args.allow_unverified,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1

    print(json.dumps(bundle.stage_payload(), indent=2, sort_keys=True))
    return 0 if bundle.verified or args.allow_unverified else 1


if __name__ == "__main__":
    raise SystemExit(main())
