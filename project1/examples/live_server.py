"""Live inference server for the Distillery demo UI (apps/distillery-demo).

Implements the demo app's serving contract with the two real local models:
  model arm student_base -> Qwen2.5-0.5B with the LoRA adapter disabled
  any other arm          -> the distilled adapter enabled

Run:
    PYTHONPATH=.:src .venv/bin/uvicorn examples.live_server:app --port 8020

Then start the demo app with:
    NEXT_PUBLIC_DISTILLERY_INFERENCE_URL=http://localhost:8020 pnpm dev
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from distillery.data.validate import parse_json
from distillery.synthesis.teacher import TASK_INSTRUCTIONS, SYSTEM_PROMPT
from distillery.contracts.dataset import canonical_json
from examples.local_distill_bedrock import ROOT

app = FastAPI(title="Distillery live serving")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
_state: dict = {}


@app.on_event("startup")
def load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).to(device)
    model = PeftModel.from_pretrained(
        model, str(ROOT / "baseline" / "model" / "adapter")).to(device)
    model.eval()
    _state.update(torch=torch, tok=tok, model=model, device=device)


class InferRequest(BaseModel):
    model_id: str
    artifact_id: str | None = None
    task: str
    example_id: str | None = None
    input: dict[str, Any]


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@app.get("/v1/demo/health")
def health():
    return {"serving_ready": bool(_state), "endpoint_id": "local-mps",
            "available_model_ids": ["student_base", "sequence_kd"]}


@app.post("/v1/demo/infer")
def infer(req: InferRequest):
    torch, tok, model = _state["torch"], _state["tok"], _state["model"]
    distilled = "student_base" not in req.model_id
    instruction = TASK_INSTRUCTIONS.get(req.task, "")
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": instruction + "\n\nINPUT:\n" + canonical_json(req.input)}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(_state["device"])
    t0 = time.time()
    ctx = _null() if distilled else model.disable_adapter()
    with ctx, torch.no_grad():
        out = model.generate(**ids, max_new_tokens=400, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    completion = out[0][ids["input_ids"].shape[1]:]
    text = tok.decode(completion, skip_special_tokens=True)
    obj = parse_json(text)
    structured = obj if isinstance(obj, dict) else {"raw_text": text[:1200]}
    return {"structured_output": structured,
            "latency_ms": int((time.time() - t0) * 1000),
            "prompt_tokens": int(ids["input_ids"].shape[1]),
            "completion_tokens": int(completion.shape[0])}
