#!/usr/bin/env python3
"""Synthesize sealed smoke train/validation teacher responses and upload to S3."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import boto3
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from distillery.backends.s3 import ImmutableS3Store, parse_s3_uri  # noqa: E402
from distillery.contracts.hashing import content_sha256, sha256_hex  # noqa: E402
from experiments.aws_smoke.model_evidence import (  # noqa: E402
    collect_tokenizer_runtime_evidence,
)
from experiments.aws_smoke.teacher_synthesis import (  # noqa: E402
    DISTILLERY_BUCKET,
    EXPECTED_MODEL_MATERIALIZATION_SHA256,
    EXPECTED_TEACHER_WEIGHT_SHA256,
    MODEL_MATERIALIZATION_URI,
    TEACHER_MODEL_ID,
    TEACHER_REVISION,
    GenerationConfig,
    build_materialization_manifest,
    build_teacher_prompt,
    file_sha256,
    generate_one,
    load_smoke_train_validation_rows,
    seal_teacher_row,
    select_device,
    set_generation_determinism,
    sync_teacher_snapshot_from_materialization,
    unique_synthesis_prefix,
    verify_model_materialization_bytes,
    write_jsonl,
)


def _download_s3_file(client: object, uri: str, dest: Path) -> None:
    parsed = parse_s3_uri(uri)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    client.download_file(parsed.bucket, parsed.key, str(tmp))
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(prog="synthesize_teacher_responses")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/teacher_synthesis_smoke_v1"),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("artifacts/models"),
    )
    parser.add_argument(
        "--aws-profile",
        default="gabriel-cli",
    )
    parser.add_argument(
        "--aws-region",
        default="us-east-1",
    )
    parser.add_argument(
        "--s3-prefix",
        default=None,
        help="Optional s3://distillery-.../synthesis/<unique> prefix",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Write local sealed artifacts only",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-split cap for dry runs (forbidden for sealed upload)",
    )
    args = parser.parse_args()
    if args.limit is not None and not args.skip_upload:
        raise SystemExit("--limit cannot be used when uploading sealed artifacts")

    started = time.perf_counter()
    session = boto3.Session(profile_name=args.aws_profile, region_name=args.aws_region)
    s3_client = session.client("s3")
    store = ImmutableS3Store(s3_client, default_bucket=DISTILLERY_BUCKET)

    mat_bytes = store.get_bytes(
        MODEL_MATERIALIZATION_URI,
        expected_sha256=EXPECTED_MODEL_MATERIALIZATION_SHA256,
    )
    materialization = verify_model_materialization_bytes(mat_bytes)

    snapshot = sync_teacher_snapshot_from_materialization(
        materialization=materialization,
        models_dir=args.models_dir,
        download_file=lambda uri, dest: _download_s3_file(s3_client, uri, dest),
    )
    weight_sha = file_sha256(snapshot / "model.safetensors")
    if weight_sha != EXPECTED_TEACHER_WEIGHT_SHA256:
        raise RuntimeError("teacher weight hash failed after sync")

    generation_config = GenerationConfig()
    set_generation_determinism(generation_config.seed)
    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    runtime = collect_tokenizer_runtime_evidence(snapshot, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype="auto",
    )
    model.to(device)
    model.eval()

    split_rows = load_smoke_train_validation_rows()
    corpus_content_sha256 = content_sha256(
        [row for split in ("train", "validation") for row in split_rows[split]]
    )
    sealed_rows = []
    for split in ("train", "validation"):
        rows = split_rows[split]
        if args.limit is not None:
            rows = rows[: args.limit]
        for index, row in enumerate(rows):
            prompt = build_teacher_prompt(row)
            raw, prompt_tokens, completion_tokens = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt,
                device=device,
                generation_config=generation_config,
            )
            sealed = seal_teacher_row(
                row=row,
                split=split,
                prompt_text=prompt,
                raw_response=raw,
                tokenizer=tokenizer,
                tokenizer_sha256=runtime.tokenizer_sha256,
                chat_template_sha256_value=runtime.chat_template_sha256,
                teacher_weight_sha256=weight_sha,
                generation_config=generation_config,
                prompt_token_count=prompt_tokens,
                generation_completion_token_count=completion_tokens,
            )
            sealed_rows.append(sealed)
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                print(
                    json.dumps(
                        {
                            "progress": True,
                            "split": split,
                            "completed": index + 1,
                            "total": len(rows),
                            "accepted": sealed.accepted,
                            "example_id": sealed.example_id,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Dataset channel files retain full envelopes for oracle arms; teacher never saw labels.
    file_sha256_map: dict[str, str] = {}
    for split in ("train", "validation"):
        path = args.output_dir / f"{split}.jsonl"
        file_sha256_map[path.name] = write_jsonl(path, split_rows[split])

    teacher_responses = [
        {
            "example_id": row.example_id,
            "split": row.split,
            "task": row.task,
            "response_text": row.teacher_payload["response_text"],
            "label_source": "teacher",
            "accepted": row.accepted,
            "rejection_reasons": list(row.rejection_reasons),
            "record_sha256": row.record.record_sha256,
        }
        for row in sealed_rows
    ]
    file_sha256_map["teacher_responses.jsonl"] = write_jsonl(
        args.output_dir / "teacher_responses.jsonl",
        teacher_responses,
    )
    file_sha256_map["response_records.jsonl"] = write_jsonl(
        args.output_dir / "response_records.jsonl",
        [row.record.model_dump(mode="json") for row in sealed_rows],
    )
    file_sha256_map["synthesis_rows.jsonl"] = write_jsonl(
        args.output_dir / "synthesis_rows.jsonl",
        [row.teacher_payload for row in sealed_rows],
    )
    rejected = [row.teacher_payload for row in sealed_rows if not row.accepted]
    file_sha256_map["rejected.jsonl"] = write_jsonl(
        args.output_dir / "rejected.jsonl",
        rejected,
    )

    s3_prefix = args.s3_prefix or unique_synthesis_prefix()
    prefix_key = parse_s3_uri(s3_prefix.rstrip("/") + "/placeholder").key.rsplit("/", 1)[
        0
    ]
    synthesis_id = prefix_key.split("/", 1)[1]
    elapsed = time.perf_counter() - started
    manifest = build_materialization_manifest(
        synthesis_id=synthesis_id,
        s3_prefix=s3_prefix,
        rows=sealed_rows,
        file_sha256_map=file_sha256_map,
        tokenizer_sha256=runtime.tokenizer_sha256,
        chat_template_digest=runtime.chat_template_sha256,
        teacher_weight_sha256=weight_sha,
        generation_config=generation_config,
        device=device,
        runtime_seconds=elapsed,
        cost_usd=0.0,
        corpus_content_sha256=corpus_content_sha256,
    )
    manifest_path = args.output_dir / "materialization.json"
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    manifest_sha = sha256_hex(manifest_path.read_bytes())
    file_sha256_map["materialization.json"] = manifest_sha
    # Bind aggregate file digests after the manifest bytes are final.
    manifest["file_sha256"] = dict(sorted(file_sha256_map.items()))
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    manifest_sha = sha256_hex(manifest_path.read_bytes())
    file_sha256_map["materialization.json"] = manifest_sha
    # Final rewrite keeps materialization.json digest consistent with on-disk bytes
    # by storing digests of peer files only under file_sha256 and reporting the
    # manifest digest separately for launchers.
    peer_hashes = {
        name: digest
        for name, digest in file_sha256_map.items()
        if name != "materialization.json"
    }
    manifest["file_sha256"] = dict(sorted(peer_hashes.items()))
    manifest["materialization_sha256"] = None
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    provisional = sha256_hex(manifest_text.encode("utf-8"))
    manifest["materialization_sha256"] = provisional
    # content hash is over peer files + body without self digest field
    body_without_self = {
        key: value for key, value in manifest.items() if key != "materialization_sha256"
    }
    body_sha = content_sha256(body_without_self)
    manifest["materialization_sha256"] = body_sha
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    manifest_sha = sha256_hex(manifest_path.read_bytes())

    uploaded: dict[str, str] = {}
    if not args.skip_upload:
        for name in (
            "train.jsonl",
            "validation.jsonl",
            "teacher_responses.jsonl",
            "response_records.jsonl",
            "synthesis_rows.jsonl",
            "rejected.jsonl",
            "materialization.json",
        ):
            path = args.output_dir / name
            digest = file_sha256(path)
            ref = store.put_file(
                f"{prefix_key}/{name}",
                path,
                content_type=(
                    "application/json"
                    if name.endswith(".json")
                    else "application/x-ndjson"
                ),
                metadata={
                    "immutable": "true",
                    "synthesis_id": synthesis_id,
                    "label_source": "teacher",
                },
            )
            if ref.sha256 != digest:
                raise RuntimeError(f"upload hash mismatch for {name}")
            # Re-download verify
            remote = store.get_bytes(ref.uri, expected_sha256=digest)
            if sha256_hex(remote) != digest:
                raise RuntimeError(f"remote verify failed for {name}")
            uploaded[name] = ref.uri

    summary = {
        "ok": True,
        "s3_prefix": s3_prefix.rstrip("/") + "/",
        "materialization_sha256": manifest_sha,
        "counts": manifest["counts"],
        "runtime_seconds": elapsed,
        "cost_usd": 0.0,
        "device": device,
        "uploaded": uploaded,
        "teacher_model_id": TEACHER_MODEL_ID,
        "teacher_revision": TEACHER_REVISION,
        "teacher_weight_sha256": weight_sha,
        "model_materialization_sha256": EXPECTED_MODEL_MATERIALIZATION_SHA256,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
