#!/usr/bin/env python3
"""Plan or render ephemeral high-bandwidth materialization of Qwen2.5-72B-Instruct.

Default mode is plan-only. Real launch requires operator approval after check_gates
reports may_execute=true. Credentials are never printed. Hard cap: $500.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.qwen72b_fallback.cost import (  # noqa: E402
    TRANSFER_HARD_CAP_USD,
    TRANSFER_HOURLY_USD,
    TRANSFER_PRICE_SOURCE,
    build_cost_artifact,
)
from experiments.qwen72b_fallback.pins import (  # noqa: E402
    DISTILLERY_BUCKET,
    MODEL_ID,
    REVISION,
    SNAPSHOT_S3_URI,
    load_weight_inventory,
    sealed_identity,
)

STATUS_KEY = "models/_ephemeral-transfer/72b-status.json"
WORKER_KEY = "models/_ephemeral-transfer/transfer_72b.py"
MAX_TRANSFER_SECONDS = 3 * 3600
INSTANCE_TYPE = "c5n.9xlarge"


def build_transfer_plan() -> dict[str, Any]:
    identity = sealed_identity()
    inventory = load_weight_inventory()
    cost = build_cost_artifact(
        kind="materialization",
        max_runtime_seconds=MAX_TRANSFER_SECONDS,
        hourly_usd=TRANSFER_HOURLY_USD,
        price_source=TRANSFER_PRICE_SOURCE,
        hard_cap_usd=TRANSFER_HARD_CAP_USD,
        instance_type=INSTANCE_TYPE,
    )
    return {
        "schema_version": "distillery.qwen72b_fallback.materialization_plan.v1",
        "mode_default": "plan",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "bucket": DISTILLERY_BUCKET,
        "destination_prefix": SNAPSHOT_S3_URI,
        "ephemeral_status_key": f"s3://{DISTILLERY_BUCKET}/{STATUS_KEY}",
        "worker_key": f"s3://{DISTILLERY_BUCKET}/{WORKER_KEY}",
        "instance_type": INSTANCE_TYPE,
        "transfer_path": "ephemeral_ec2_hf_transfer_then_multipart_s3",
        "verify_checksums": True,
        "versioned_manifest": (
            "snapshot-manifest.json + SHA256SUMS + models/materialization.json merge"
        ),
        "cleanup": "terminate EC2 + delete ephemeral local disk; retain sealed S3 snapshot",
        "hard_cap_usd": TRANSFER_HARD_CAP_USD,
        "cost": cost,
        "identity": identity.model_dump(mode="json"),
        "inventory_sha256": inventory["inventory_sha256"],
        "n_files": len(inventory["files"]),
        "total_safetensors_bytes": inventory["total_safetensors_bytes"],
        "notes": (
            "Launch only after check_gates.py --action materialize reports "
            "may_execute=true. Do not start while 14B/32B transfer or g5 smoke is active."
        ),
        "planned_at_utc": datetime.now(UTC).isoformat(),
    }


def render_worker_script() -> str:
    inventory = load_weight_inventory()
    expected = {
        name: [meta["sha256"], meta["size"]] for name, meta in inventory["files"].items()
    }
    plan = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "bucket": DISTILLERY_BUCKET,
        "prefix": f"models/Qwen/Qwen2.5-72B-Instruct/{REVISION}",
        "status_key": STATUS_KEY,
        "expected": expected,
        "hard_cap_usd": TRANSFER_HARD_CAP_USD,
        "hourly_usd": TRANSFER_HOURLY_USD,
    }
    plan_literal = json.dumps(plan, indent=2, sort_keys=True)
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        \"\"\"Ephemeral 72B materializer. Generated; do not embed credentials.\"\"\"
        from __future__ import annotations

        import hashlib
        import json
        import os
        import time
        from datetime import datetime, timezone
        from pathlib import Path

        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config
        from huggingface_hub import snapshot_download

        PLAN = json.loads('''{plan_literal}''')
        BUCKET = PLAN["bucket"]
        PREFIX = PLAN["prefix"]
        STATUS_KEY = PLAN["status_key"]
        EXPECTED = {{name: (digest, size) for name, (digest, size) in PLAN["expected"].items()}}
        WORK = Path("/var/tmp/distillery-model-transfer/qwen72b")
        REGION = "us-east-1"
        S3 = boto3.client(
            "s3",
            region_name=REGION,
            config=Config(
                retries={{"max_attempts": 10, "mode": "adaptive"}},
                max_pool_connections=64,
            ),
        )
        TRANSFER = TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            max_concurrency=8,
            multipart_chunksize=64 * 1024 * 1024,
        )
        START = time.time()


        def put_status(payload):
            body = {{
                **payload,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.time() - START,
            }}
            S3.put_object(
                Bucket=BUCKET,
                Key=STATUS_KEY,
                Body=(json.dumps(body, indent=2, sort_keys=True) + "\\n").encode(),
                ContentType="application/json",
            )


        def sha256_file(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()


        def assert_under_cap() -> None:
            hours = (time.time() - START) / 3600.0
            if float(PLAN["hourly_usd"]) * hours > float(PLAN["hard_cap_usd"]):
                raise SystemExit("hard cap exceeded")


        def main() -> None:
            put_status(
                {{
                    "phase": "download",
                    "ok": False,
                    "model_id": PLAN["model_id"],
                    "revision": PLAN["revision"],
                }}
            )
            assert_under_cap()
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            local = Path(
                snapshot_download(
                    repo_id=PLAN["model_id"],
                    revision=PLAN["revision"],
                    local_dir=str(WORK),
                    local_dir_use_symlinks=False,
                    ignore_patterns=[".gitattributes"],
                )
            )
            put_status({{"phase": "verify", "ok": False}})
            checksums = {{}}
            for name, (digest, size) in EXPECTED.items():
                path = local / name
                if not path.is_file() or path.stat().st_size != size:
                    raise SystemExit(f"missing or size mismatch: {{name}}")
                actual = sha256_file(path)
                if actual != digest:
                    raise SystemExit(f"checksum mismatch: {{name}}")
                checksums[name] = actual
                assert_under_cap()
            put_status({{"phase": "upload", "ok": False, "verified_files": len(checksums)}})
            for name in sorted(EXPECTED):
                key = f"{{PREFIX}}/{{name}}"
                S3.upload_file(
                    str(local / name),
                    BUCKET,
                    key,
                    ExtraArgs={{
                        "Metadata": {{
                            "sha256": checksums[name],
                            "immutable": "true",
                            "model_id": PLAN["model_id"],
                            "revision": PLAN["revision"],
                        }}
                    }},
                    Config=TRANSFER,
                )
                put_status({{"phase": "upload", "ok": False, "completed_file": name}})
                assert_under_cap()
            sums = (
                "\\n".join(
                    f"{{digest}}  {{name}}" for name, digest in sorted(checksums.items())
                )
                + "\\n"
            )
            S3.put_object(
                Bucket=BUCKET,
                Key=f"{{PREFIX}}/SHA256SUMS",
                Body=sums.encode(),
                ContentType="text/plain",
            )
            manifest = {{
                "schema_version": "distillery.model_snapshot.v1",
                "model_id": PLAN["model_id"],
                "revision": PLAN["revision"],
                "files": checksums,
                "materialized_at_utc": datetime.now(timezone.utc).isoformat(),
            }}
            S3.put_object(
                Bucket=BUCKET,
                Key=f"{{PREFIX}}/snapshot-manifest.json",
                Body=(json.dumps(manifest, indent=2, sort_keys=True) + "\\n").encode(),
                ContentType="application/json",
            )
            put_status({{"phase": "done", "ok": True, "verified_files": len(checksums)}})


        if __name__ == "__main__":
            main()
        """
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="materialize_to_s3")
    parser.add_argument("--mode", choices=("plan", "render-worker"), default="plan")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Safety latch: always refused in-repo; use operator path after gates pass.",
    )
    args = parser.parse_args()
    if args.execute:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        "execute refused by script safety latch; run check_gates.py and "
                        "only launch from an operator-approved path when may_execute=true "
                        "and no 14B/g5 work is active"
                    ),
                },
                indent=2,
            )
        )
        return 2
    if args.mode == "plan":
        print(json.dumps(build_transfer_plan(), indent=2, sort_keys=True))
        return 0
    print(render_worker_script())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
