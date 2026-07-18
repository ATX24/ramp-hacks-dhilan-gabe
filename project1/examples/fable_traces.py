"""Generate a large teacher-trace set from Claude Fable for the finance world.

Labels train+validation examples from the FULL corpus with claude-fable-5,
validates each response deterministically, and writes SageMaker-format traces
(prompt/response pairs) plus an S3 upload.

Usage: ANTHROPIC_API_KEY=... PYTHONPATH=.:src .venv/bin/python examples/fable_traces.py <n_traces>
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from distillery.contracts.dataset import Example, canonical_json
from distillery.data.generate import generate_corpus, FULL_PLAN
from distillery.data.validate import validate_output
from distillery.synthesis.teacher import build_prompt, SYSTEM_PROMPT

MODEL = "claude-fable-5"
ROOT = Path(__file__).resolve().parent.parent / "runs_local"


def main(n: int):
    import anthropic
    client = anthropic.Anthropic()

    gen_dir = ROOT / "corpus_full"
    if not (gen_dir / "train.jsonl").exists():
        print("generating full synthetic corpus...")
        generate_corpus(FULL_PLAN, gen_dir)
    examples: list[Example] = []
    for split in ("train", "validation"):
        for l in (gen_dir / f"{split}.jsonl").read_text().splitlines():
            examples.append(Example.model_validate_json(l))
    examples = examples[:n]
    print(f"labeling {len(examples)} examples with {MODEL} (24 workers)...")

    def call(ex: Example) -> str:
        r = client.messages.create(
            model=MODEL, max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(ex)}])
        return next((b.text for b in r.content if b.type == "text"), "")

    with ThreadPoolExecutor(max_workers=24) as pool:
        texts = list(pool.map(call, examples))

    traces, invalid = [], 0
    for ex, text in zip(examples, texts):
        obj, errs = validate_output(ex.task, text, ex.input)
        if errs:
            invalid += 1
            continue
        traces.append({"prompt": build_prompt(ex), "response": canonical_json(obj),
                       "task": ex.task, "teacher": MODEL})
    out = ROOT / "fable_traces.jsonl"
    out.write_text("\n".join(json.dumps(t) for t in traces))
    print(json.dumps({"requested": len(examples), "valid_traces": len(traces),
                      "invalid": invalid, "file": str(out)}, indent=2))

    import boto3
    s3 = boto3.Session(profile_name="ramp-hackathon", region_name="us-east-1").client("s3")
    s3.upload_file(str(out), "proof-ramp-hackathon-225989358036",
                   "sagemaker/fable/train.jsonl")
    print("uploaded to s3://proof-ramp-hackathon-225989358036/sagemaker/fable/train.jsonl")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
