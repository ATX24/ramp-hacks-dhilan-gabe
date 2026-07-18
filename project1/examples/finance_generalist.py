"""End-to-end TinyFable happy path against a Distillery deployment.

Usage: python examples/finance_generalist.py [BASE_URL]
No training starts unless you pass --go after synthesis is real.
"""
import sys

from distillery_sdk import Distillery

base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
d = Distillery(base_url=base)

print("recipes:", [r["name"] for r in d.recipes()["recipes"] if r["implemented"]])

dataset = d.datasets.generate(corpus="smoke")
print("dataset:", dataset["dataset_id"], dataset["counts_by_split"])

plan = d.plan(dataset, recipe="auto")
print("resolved recipe:", plan["recipe"]["resolved"], "| reasons:", plan["recipe"]["resolver_reasons"])
print("teacher:", plan["models"]["teacher"]["id"], "| est cost:", plan["estimates"])
print("blockers:", plan["blockers"])

# Dry-run synthesis: counts Opus calls needed, spends nothing.
stats = d.datasets.synthesize(dataset["dataset_id"], mode="teacher", dry_run=True)
print("synthesis dry-run:", stats)
