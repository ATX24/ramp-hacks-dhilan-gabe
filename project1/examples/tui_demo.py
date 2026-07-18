"""Distillery TUI: base vs distilled, side by side, live on your machine.

One 0.5B model in memory; the LoRA adapter toggles off (base) and on
(distilled) per request. Every output is validated against the executable
oracle. A running scoreboard accumulates as you try held-out tasks.

    PYTHONPATH=.:src .venv/bin/python examples/tui_demo.py

Commands: a task number, "n" for next, "r" for a random task, "q" to quit.
"""
from __future__ import annotations

import json
import random
import time

from rich.console import Console, Group
from rich.columns import Columns
from rich.json import JSON
from rich.layout import Layout
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from distillery.contracts.dataset import Example, canonical_json
from distillery.data.validate import validate_output
from distillery.synthesis.teacher import build_prompt, SYSTEM_PROMPT
from examples.local_distill_bedrock import ROOT, load_corpus

KEY_FIELDS = {
    "transaction_review": ["gl_account", "policy_action", "journal_entry"],
    "variance_analysis": ["profit_impact_minor", "direction", "top_drivers"],
    "cash_reconciliation": ["status", "difference_minor"],
}

console = Console()


def load_models():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    with console.status("[bold]loading Qwen2.5-0.5B + distilled adapter..."):
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).to(device)
        model = PeftModel.from_pretrained(
            model, str(ROOT / "baseline" / "model" / "adapter")).to(device)
        model.eval()
    return torch, tok, model, device


def generate(torch, tok, model, device, ex: Example, distilled: bool) -> dict:
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": build_prompt(ex)}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    ctx = model.disable_adapter() if not distilled else _null()
    with ctx, torch.no_grad():
        out = model.generate(**ids, max_new_tokens=400, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    obj, errs = validate_output(ex.task, text, ex.input)
    keys = KEY_FIELDS[ex.task]
    correct = (not errs) and all(obj.get(k) == ex.expected_output.get(k) for k in keys)
    return {"text": text, "obj": obj, "valid": not errs, "errors": errs,
            "correct": correct, "secs": time.time() - t0}


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def verdict_line(r: dict) -> Text:
    t = Text()
    t.append(" SCHEMA VALID " if r["valid"] else " SCHEMA INVALID ",
             style="black on green" if r["valid"] else "white on red")
    t.append("  ")
    t.append(" DECISION FIELDS OK " if r["correct"] else " DECISION FIELDS WRONG ",
             style="black on green" if r["correct"] else "white on dark_orange")
    t.append(f"   {r['secs']:.1f}s", style="dim")
    return t


def model_panel(title: str, r: dict, color: str) -> Panel:
    body = JSON(canonical_json(r["obj"])) if r["obj"] else Text(r["text"][:800], style="dim")
    return Panel(Group(verdict_line(r), Rule(style="dim"), body),
                 title=f"[bold]{title}", border_style=color, height=24)


def main():
    torch, tok, model, device = load_models()
    tests = [e for e in load_corpus()["test_iid"] if e.task in KEY_FIELDS]
    score = {"base": {"valid": 0, "correct": 0}, "dist": {"valid": 0, "correct": 0}, "n": 0}
    idx = 0

    console.print(Panel.fit(
        "[bold]DISTILLERY[/bold]  base vs distilled, same 0.5B weights\n"
        "[dim]held-out finance tasks, oracle-checked, fully local[/dim]",
        border_style="red"))

    while True:
        ex = tests[idx % len(tests)]
        console.print(Rule(f"[bold]#{idx % len(tests)}  {ex.task}  ({ex.difficulty})"))
        console.print(Panel(Text(canonical_json(ex.input)[:500] + " ...", style="dim"),
                            title="input", border_style="grey50"))

        with console.status("[bold]base model thinking (no adapter)..."):
            base = generate(torch, tok, model, device, ex, distilled=False)
        with console.status("[bold red]distilled model thinking (adapter on)..."):
            dist = generate(torch, tok, model, device, ex, distilled=True)

        console.print(Columns([
            model_panel("BASE Qwen2.5-0.5B", base, "grey50"),
            model_panel("DISTILLED student", dist, "red"),
        ], equal=True, expand=True))
        console.print(Panel(JSON(canonical_json(ex.expected_output)),
                            title="oracle expected", border_style="green", height=14))

        score["n"] += 1
        for k, r in (("base", base), ("dist", dist)):
            score[k]["valid"] += r["valid"]
            score[k]["correct"] += r["correct"]
        tbl = Table(title=f"scoreboard after {score['n']} task(s)")
        tbl.add_column("model")
        tbl.add_column("schema valid")
        tbl.add_column("decision fields")
        tbl.add_row("base", f"{score['base']['valid']}/{score['n']}",
                    f"{score['base']['correct']}/{score['n']}")
        tbl.add_row("[red]distilled", f"[red]{score['dist']['valid']}/{score['n']}",
                    f"[red]{score['dist']['correct']}/{score['n']}")
        console.print(tbl)

        cmd = console.input("[bold]next task \\[n], random \\[r], number, or quit \\[q]: ").strip().lower()
        if cmd == "q":
            break
        if cmd == "r":
            idx = random.randrange(len(tests))
        elif cmd.isdigit():
            idx = int(cmd)
        else:
            idx += 1


if __name__ == "__main__":
    main()
