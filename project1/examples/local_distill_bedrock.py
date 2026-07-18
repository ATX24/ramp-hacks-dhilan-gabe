"""Fast local distillation: Nova Pro (teacher, via Bedrock) -> Qwen2.5-0.5B (student, QLoRA on MPS).

Labels the finance corpus once with the teacher, then builds TWO datasets:
  baseline : every valid teacher label (sequence.v1 as-is)
  custom   : rejection_sampling.v1 semantics — valid AND oracle-agreeing labels only
Trains both and writes datasets + stats under runs_local/.

Usage: PYTHONPATH=.:src AWS_PROFILE=ramp-hackathon .venv/bin/python examples/local_distill_bedrock.py label
       (then) ... train baseline | train custom | eval
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3

from distillery.contracts.dataset import Example, canonical_json
from distillery.data.generate import generate_corpus, SMOKE_PLAN
from distillery.data.validate import validate_output
from distillery.synthesis.teacher import build_prompt, SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent.parent / "runs_local"
TEACHER = "amazon.nova-pro-v1:0"
N_TRAIN, N_VAL, N_TEST = 160, 40, 40

_rt = boto3.Session(profile_name="ramp-hackathon", region_name="us-east-1").client("bedrock-runtime")


def teacher_call(ex: Example) -> str:
    resp = _rt.converse(
        modelId=TEACHER,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": build_prompt(ex)}]}],
        inferenceConfig={"maxTokens": 1024, "temperature": 0},
    )
    return resp["output"]["message"]["content"][0]["text"]


def load_corpus() -> dict[str, list[Example]]:
    gen_dir = ROOT / "corpus"
    if not (gen_dir / "train.jsonl").exists():
        generate_corpus(SMOKE_PLAN, gen_dir)
    out = {}
    for split, cap in [("train", N_TRAIN), ("validation", N_VAL),
                       ("test_iid", N_TEST), ("test_ood", N_TEST)]:
        lines = (gen_dir / f"{split}.jsonl").read_text().splitlines()
        out[split] = [Example.model_validate_json(l) for l in lines[:cap]]
    return out


def label():
    corpus = load_corpus()
    todo = corpus["train"] + corpus["validation"]
    print(f"labeling {len(todo)} examples with {TEACHER} (12 workers)...")
    with ThreadPoolExecutor(max_workers=12) as pool:
        texts = list(pool.map(teacher_call, todo))

    stats = {"labeled": 0, "invalid": 0, "oracle_agree": 0}
    labeled = []
    for ex, text in zip(todo, texts):
        obj, errs = validate_output(ex.task, text, ex.input)
        rec = ex.model_dump()
        if not errs:
            rec["response"] = canonical_json(obj)
            rec["provenance"]["label_source"] = "teacher"
            rec["provenance"]["teacher_model"] = TEACHER
            stats["labeled"] += 1
            if canonical_json(obj) == canonical_json(ex.expected_output):
                stats["oracle_agree"] += 1
                rec["_oracle_agree"] = True
        else:
            stats["invalid"] += 1
            rec["response"] = None
        labeled.append(rec)

    ROOT.mkdir(exist_ok=True)
    (ROOT / "labeled.jsonl").write_text("\n".join(json.dumps(r) for r in labeled))
    base = [dict(r, **{}) for r in labeled if r.get("response")]
    cust = [r for r in base if r.get("_oracle_agree")]
    for r in base + cust:
        r.pop("_oracle_agree", None)
    (ROOT / "baseline.jsonl").write_text("\n".join(json.dumps(r) for r in base))
    (ROOT / "custom.jsonl").write_text("\n".join(json.dumps(r) for r in cust))
    stats["baseline_n"], stats["custom_n"] = len(base), len(cust)
    (ROOT / "label_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def make_manifest(run: str) -> Path:
    from distillery.contracts.manifest import (RunManifest, DatasetRef, ModelPin,
                                               TrainingConfig, RuntimeConfig, CostConfig)
    m = RunManifest(
        run_id=f"local-{run}", created_at="2026-07-18T00:00:00Z",
        dataset=DatasetRef(dataset_id=f"local-{run}", uri=f"file://runs_local/{run}.jsonl",
                           sha256="local", split_sha256={}),
        models={"teacher": ModelPin(id=TEACHER, revision=TEACHER, access="api_black_box"),
                "student": ModelPin(id="Qwen/Qwen2.5-0.5B-Instruct", revision="main",
                                    access="white_box")},
        recipe={"requested": run, "resolved": "sequence.v1", "resolver_reasons": [],
                "rejected_alternatives": []},
        arm="sequence_kd", training=TrainingConfig(seed=17, max_steps=150),
        runtime=RuntimeConfig(backend="local"), cost=CostConfig(max_run_usd=5.0),
        output_prefix=f"file://runs_local/{run}/")
    p = ROOT / f"{run}.manifest.json"
    p.write_text(m.model_dump_json())
    return p


def train(run: str):
    from distillery.training.entrypoint import run as train_run
    metrics = train_run(make_manifest(run), ROOT / f"{run}.jsonl", ROOT / run)
    print(json.dumps(metrics, indent=2))


def evaluate():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    corpus = load_corpus()
    tests = corpus["test_iid"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    def score(adapter: str | None) -> dict:
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                                     torch_dtype=torch.float32).to(device)
        if adapter:
            model = PeftModel.from_pretrained(model, adapter).to(device)
        model.eval()
        valid = agree = 0
        for ex in tests:
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": build_prompt(ex)}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=512, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            obj, errs = validate_output(ex.task, text, ex.input)
            if not errs:
                valid += 1
                keys = {"transaction_review": ["gl_account", "policy_action", "journal_entry"],
                        "variance_analysis": ["profit_impact_minor", "direction", "top_drivers"],
                        "cash_reconciliation": ["status", "difference_minor"],
                        "merchant_tagging": ["merchant", "category"]}[ex.task]
                if all(obj.get(k) == ex.expected_output.get(k) for k in keys):
                    agree += 1
        del model
        return {"n": len(tests), "schema_valid": valid / len(tests),
                "key_field_accuracy": agree / len(tests)}

    results = {"base_untrained": score(None),
               "baseline_distilled": score(str(ROOT / "baseline" / "model" / "adapter")),
               "custom_distilled": score(str(ROOT / "custom" / "model" / "adapter"))}
    (ROOT / "eval.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "label":
        label()
    elif cmd == "train":
        train(sys.argv[2])
    elif cmd == "eval":
        evaluate()
