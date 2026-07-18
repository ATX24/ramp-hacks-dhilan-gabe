"""Side-by-side demo: base Qwen2.5-0.5B vs the distilled adapter on finance tasks.

One model in memory; the LoRA adapter is toggled off (base) / on (distilled)
per request. Run:

    PYTHONPATH=.:src .venv/bin/uvicorn examples.compare_demo:app --port 8010
    open http://localhost:8010

Env: DEMO_ADAPTER=runs_local/custom/model/adapter (default: custom, else baseline)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from distillery.contracts.dataset import Example, canonical_json
from distillery.data.validate import validate_output
from distillery.synthesis.teacher import build_prompt, SYSTEM_PROMPT
from examples.local_distill_bedrock import ROOT, load_corpus

KEY_FIELDS = {
    "transaction_review": ["gl_account", "policy_action", "journal_entry"],
    "variance_analysis": ["profit_impact_minor", "direction", "top_drivers"],
    "cash_reconciliation": ["status", "difference_minor"],
    "merchant_tagging": ["merchant", "category"],
}

app = FastAPI(title="Distillery demo — base vs distilled")
_state: dict = {}


def _adapter_dir() -> Path:
    if os.environ.get("DEMO_ADAPTER"):
        return Path(os.environ["DEMO_ADAPTER"])
    for run in ("custom", "baseline"):
        p = ROOT / run / "model" / "adapter"
        if p.exists():
            return p
    raise FileNotFoundError("No trained adapter found under runs_local/; train first.")


@app.on_event("startup")
def load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                                 torch_dtype=torch.float32).to(device)
    adapter = _adapter_dir()
    model = PeftModel.from_pretrained(model, str(adapter)).to(device)
    model.eval()
    tests = load_corpus()["test_iid"]
    _state.update(model=model, tok=tok, device=device, tests=tests,
                  adapter=str(adapter), torch=torch)


def _generate(ex: Example, distilled: bool) -> dict:
    torch, model, tok = _state["torch"], _state["model"], _state["tok"]
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": build_prompt(ex)}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(_state["device"])
    t0 = time.time()
    ctx = model.disable_adapter() if not distilled else _nullcontext()
    with ctx, torch.no_grad():
        out = model.generate(**ids, max_new_tokens=512, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    obj, errs = validate_output(ex.task, text, ex.input)
    keys = KEY_FIELDS[ex.task]
    key_ok = (not errs) and all(obj.get(k) == ex.expected_output.get(k) for k in keys)
    return {"raw": text[:2000], "parsed": obj, "schema_valid": not errs,
            "validation_errors": errs[:5], "key_fields_correct": key_ok,
            "latency_s": round(time.time() - t0, 2)}


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@app.get("/api/examples")
def list_examples():
    return [{"i": i, "task": ex.task, "difficulty": ex.difficulty,
             "preview": canonical_json(ex.input)[:120]}
            for i, ex in enumerate(_state["tests"])]


class RunReq(BaseModel):
    i: int


@app.post("/api/run")
def run(req: RunReq):
    ex: Example = _state["tests"][req.i]
    return {"task": ex.task, "difficulty": ex.difficulty,
            "input": ex.input, "expected": ex.expected_output,
            "adapter": _state["adapter"],
            "base": _generate(ex, distilled=False),
            "distilled": _generate(ex, distilled=True)}


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Distillery — base vs distilled</title><style>
body{font-family:Georgia,serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}
h1{font-size:1.4rem;border-bottom:1px solid #ccc;padding-bottom:.5rem}
select,button{font:inherit;padding:.4rem .8rem;margin-right:.5rem}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem}
.card{border:1px solid #ddd;padding:1rem;border-radius:6px}
.card h2{font-size:1rem;margin:0 0 .5rem}
pre{background:#f7f7f5;padding:.7rem;overflow-x:auto;font-size:.78rem;white-space:pre-wrap}
.badge{display:inline-block;padding:.1rem .5rem;border-radius:9px;font-size:.75rem;
 font-family:monospace;margin-right:.4rem}
.ok{background:#e2f2e4;color:#186b26}.bad{background:#fae3e3;color:#8f1d1d}
.meta{color:#777;font-size:.8rem}#expected{margin-top:1rem}</style></head><body>
<h1>Distillery — same 0.5B model, before and after distillation from Nova Pro</h1>
<div><select id="sel"></select><button onclick="go()">Run both</button>
<span id="status" class="meta"></span></div>
<div class="cols">
<div class="card"><h2>Base Qwen2.5-0.5B (not distilled)</h2><div id="b_badges"></div><pre id="b_out">—</pre></div>
<div class="card"><h2>Distilled student (LoRA adapter)</h2><div id="d_badges"></div><pre id="d_out">—</pre></div>
</div>
<div class="card" id="expected"><h2>Oracle expected output</h2><pre id="e_out">—</pre></div>
<script>
async function init(){
  const ex = await (await fetch('/api/examples')).json();
  sel.innerHTML = ex.map(e=>`<option value="${e.i}">#${e.i} ${e.task} (${e.difficulty}) — ${e.preview}…</option>`).join('');
}
function badges(r){
  return `<span class="badge ${r.schema_valid?'ok':'bad'}">schema ${r.schema_valid?'valid':'INVALID'}</span>`+
         `<span class="badge ${r.key_fields_correct?'ok':'bad'}">decision fields ${r.key_fields_correct?'correct':'wrong'}</span>`+
         `<span class="badge">${r.latency_s}s</span>`;
}
async function go(){
  status.textContent=' running…';
  const r = await (await fetch('/api/run',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({i:+sel.value})})).json();
  b_badges.innerHTML=badges(r.base); d_badges.innerHTML=badges(r.distilled);
  b_out.textContent=r.base.parsed?JSON.stringify(r.base.parsed,null,1):r.base.raw;
  d_out.textContent=r.distilled.parsed?JSON.stringify(r.distilled.parsed,null,1):r.distilled.raw;
  e_out.textContent=JSON.stringify(r.expected,null,1);
  status.textContent=` adapter: ${r.adapter}`;
}
init();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
