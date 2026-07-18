"""One-job trainer entrypoint. Consumes a sealed manifest, materializes inputs,
runs completion-only QLoRA SFT (sequence.v1) on the TinyFable student, evaluates
run-local checks, writes artifacts + checksums, and exits.

Heavy ML deps are imported lazily so the control plane / API image never needs
torch. Invoke:  python -m distillery.training.entrypoint --manifest manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ..contracts.manifest import RunManifest
from ..contracts.errors import RecipeNotImplemented, ArtifactIntegrityFailed

STUDENT_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"


def load_examples(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_chat(example: dict, tokenizer) -> tuple[str, str]:
    from ..synthesis.teacher import build_prompt, SYSTEM_PROMPT
    from ..contracts.dataset import Example
    ex = Example.model_validate(example)
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": build_prompt(ex)}],
        tokenize=False, add_generation_prompt=True)
    return prompt, ex.response or ""


def run(manifest_path: Path, data_path: Path, out_dir: Path, dry_run: bool = False) -> dict:
    manifest = RunManifest.model_validate_json(manifest_path.read_text())
    if manifest.recipe.get("resolved") != "sequence.v1":
        raise RecipeNotImplemented(
            f"This entrypoint executes sequence.v1 only; got {manifest.recipe.get('resolved')}. "
            "logit.v1 requires the white-box local pair and is gated separately.")

    examples = load_examples(data_path)
    missing = [e["example_id"] for e in examples if not e.get("response")]
    if missing:
        raise ArtifactIntegrityFailed(f"{len(missing)} training examples lack responses; run synthesis first.",
                                      details={"first": missing[:5]})
    if dry_run:
        return {"ok": True, "dry_run": True, "n_examples": len(examples),
                "student": manifest.models["student"].id, "seed": manifest.training.seed}

    # ---- heavy imports only past this point ----
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model

    cfg = manifest.training
    torch.manual_seed(cfg.seed)
    student_id = manifest.models["student"].id
    revision = manifest.models["student"].revision
    tokenizer = AutoTokenizer.from_pretrained(student_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        student_id, revision=revision,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model.gradient_checkpointing_enable()
    q = cfg.qlora
    model = get_peft_model(model, LoraConfig(
        r=q["r"], lora_alpha=q["alpha"], lora_dropout=q["dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))

    class SFTDataset(torch.utils.data.Dataset):
        def __init__(self, exs):
            self.exs = exs
        def __len__(self):
            return len(self.exs)
        def __getitem__(self, i):
            prompt, completion = build_chat(self.exs[i], tokenizer)
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tokenizer(completion + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            c_ids = c_ids[: cfg.completion_cap]
            ids = (p_ids + c_ids)[-cfg.max_length:]
            labels = [-100] * max(0, len(ids) - len(c_ids)) + c_ids[-len(ids):]
            return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels[: len(ids)]),
                    "attention_mask": torch.ones(len(ids), dtype=torch.long)}

    def collate(batch):
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id
        n = max(len(b["input_ids"]) for b in batch)
        out = {k: [] for k in ("input_ids", "labels", "attention_mask")}
        for b in batch:
            d = n - len(b["input_ids"])
            out["input_ids"].append(torch.cat([b["input_ids"], torch.full((d,), pad)]))
            out["labels"].append(torch.cat([b["labels"], torch.full((d,), -100)]))
            out["attention_mask"].append(torch.cat([b["attention_mask"], torch.zeros(d, dtype=torch.long)]))
        return {k: torch.stack(v) for k, v in out.items()}

    args = TrainingArguments(
        output_dir=str(out_dir / "training"), max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.micro_batch, gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr, warmup_ratio=0.05, lr_scheduler_type="cosine",
        logging_steps=5, save_strategy="no", seed=cfg.seed, report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=SFTDataset(examples), data_collator=collate)
    result = trainer.train()

    adapter_dir = out_dir / "model" / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    sums = {}
    for p in sorted(adapter_dir.rglob("*")):
        if p.is_file():
            sums[str(p.relative_to(out_dir))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out_dir / "integrity").mkdir(parents=True, exist_ok=True)
    (out_dir / "integrity" / "SHA256SUMS").write_text(
        "\n".join(f"{v}  {k}" for k, v in sums.items()))
    metrics = {"train_loss_final": result.training_loss, "steps": cfg.max_steps,
               "manifest_sha256": manifest.seal_hash()}
    (out_dir / "training" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run(Path(a.manifest), Path(a.data), Path(a.out), dry_run=a.dry_run), indent=2))


if __name__ == "__main__":
    main()
