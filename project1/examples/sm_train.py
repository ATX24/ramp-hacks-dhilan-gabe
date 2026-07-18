"""SageMaker entrypoint: fast LoRA SFT of Qwen2.5-0.5B on finance traces.

Runs inside the HuggingFace DLC. Reads train.jsonl from the SageMaker train
channel, trains bf16 with a large batch, saves the adapter to /opt/ml/model.
"""
import json
import os
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model

STUDENT = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_STEPS = int(os.environ.get("SM_HP_MAX_STEPS", "200"))
DATA = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")) / "train.jsonl"
OUT = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

SYSTEM_PROMPT = (
    "You are a meticulous corporate finance engine. Respond with ONLY a single JSON object "
    "matching the requested task schema. No prose, no markdown fences. Amounts are integer "
    "minor units. Journal entries must balance exactly. Follow policy rule precedence."
)


def main():
    examples = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    examples = [e for e in examples if e.get("response")]
    print(f"training on {len(examples)} labeled examples, {MAX_STEPS} steps")

    tok = AutoTokenizer.from_pretrained(STUDENT)
    model = AutoModelForCausalLM.from_pretrained(STUDENT, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, i):
            e = examples[i]
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": e["prompt"]}],
                tokenize=False, add_generation_prompt=True)
            p = tok(prompt, add_special_tokens=False)["input_ids"]
            c = tok(e["response"] + tok.eos_token, add_special_tokens=False)["input_ids"][:512]
            ids = (p + c)[-2048:]
            labels = [-100] * max(0, len(ids) - len(c)) + c[-len(ids):]
            return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels[:len(ids)]),
                    "attention_mask": torch.ones(len(ids), dtype=torch.long)}

    def collate(batch):
        pad = tok.pad_token_id or tok.eos_token_id
        n = max(len(b["input_ids"]) for b in batch)
        out = {k: [] for k in ("input_ids", "labels", "attention_mask")}
        for b in batch:
            d = n - len(b["input_ids"])
            out["input_ids"].append(torch.cat([b["input_ids"], torch.full((d,), pad)]))
            out["labels"].append(torch.cat([b["labels"], torch.full((d,), -100)]))
            out["attention_mask"].append(torch.cat([b["attention_mask"],
                                                    torch.zeros(d, dtype=torch.long)]))
        return {k: torch.stack(v) for k, v in out.items()}

    args = TrainingArguments(
        output_dir="/tmp/train", max_steps=MAX_STEPS,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=2e-4, warmup_ratio=0.05, lr_scheduler_type="cosine",
        bf16=True, logging_steps=10, save_strategy="no", report_to=[], seed=17)
    result = Trainer(model=model, args=args, train_dataset=DS(),
                     data_collator=collate).train()

    model.save_pretrained(OUT / "adapter")
    tok.save_pretrained(OUT / "adapter")
    (OUT / "metrics.json").write_text(json.dumps(
        {"train_loss_final": result.training_loss, "steps": MAX_STEPS,
         "n_examples": len(examples)}))
    print("DONE", result.training_loss)


if __name__ == "__main__":
    main()
