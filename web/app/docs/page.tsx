import type { Metadata } from "next";
import Link from "next/link";
import { SiteFooter, SiteHeader } from "../components/SiteChrome";

export const metadata: Metadata = {
  title: "Docs | Distillery",
  description:
    "Distillery documentation: quickstart, the Python SDK, user-defined recipes, and the proof gate.",
};

const sections = [
  {
    id: "quickstart",
    kicker: "01 · Quickstart",
    title: "Three lines to a sealed distillation",
    body: "Point Distillery at a dataset, pick a recipe, and wait for the proof report. Everything else, teacher labeling, manifest sealing, training, evaluation, happens inside the run.",
    code: `from distillery_sdk import Distillery

distillery = Distillery(base_url="https://distillery.fly.dev")
dataset = distillery.datasets.generate(corpus="smoke")
run = distillery.distill(dataset, recipe="auto").wait()
print(run.proof_report)`,
  },
  {
    id: "datasets",
    kicker: "02 · Datasets",
    title: "Immutable, hashed, split-frozen",
    body: "Datasets are immutable resources with frozen train, validation, and test splits. Teacher labels fill only missing training responses, carry full provenance, and must pass deterministic validation. Any curation produces a new dataset id, never a mutation.",
    code: `dataset = distillery.datasets.create("./traces.jsonl")
meta = distillery.datasets.get(dataset["dataset_id"])
stats = distillery.datasets.synthesize(
    dataset["dataset_id"], mode="teacher", dry_run=True)`,
  },
  {
    id: "recipes",
    kicker: "03 · Recipes",
    title: "Bring your own distillation method",
    body: "A recipe is ordinary Python written against five synthesis primitives. Registered recipes run through the same resolver and capability gates as the built-ins. A recipe that demands something the backend cannot provide fails loudly with RECIPE_INCOMPATIBLE. It never silently downgrades.",
    code: `from distillery.recipes.custom import SynthesisContext, register

class RejectHard:
    name = "reject_hard.v1"
    requires = frozenset({"teacher"})
    description = "Drop hard cases the teacher got wrong."

    def run(self, ctx: SynthesisContext, examples):
        ctx.teacher_label(examples)
        return [ex for ex in examples
                if ctx.is_valid(ex) and ctx.agrees_with_oracle(ex)]

register(RejectHard())`,
  },
  {
    id: "runs",
    kicker: "04 · Runs",
    title: "Plan first, then one finite job",
    body: "Every run starts with a plan: resolved recipe, cost estimate, blockers. Submission seals a manifest with dataset hashes, model pins, and training config. Training executes from the manifest on local hardware or SageMaker and produces an artifact plus checksums.",
    code: `plan = distillery.plan(dataset, recipe="rejection_sampling.v1")
run = distillery.distill(
    dataset, recipe="rejection_sampling.v1", max_run_usd=25.0)
run.wait()`,
  },
  {
    id: "proof",
    kicker: "05 · The proof gate",
    title: "Pass, or BLOCKED",
    body: "Candidates are evaluated on a frozen held-out set the training process never touched: schema validity, decision-field accuracy against the executable oracle, cost ratio versus the teacher, and break-even request count. In our reference experiment the distilled 0.5B student scored 55% schema validity and 25% decision-field accuracy, equal to its teacher on decisions, at roughly a thousandth the size.",
    code: `report = run.proof_report
assert report["passed"], "gate says BLOCKED, nothing ships"`,
  },
  {
    id: "local",
    kicker: "06 · Local demos",
    title: "See it on your own machine",
    body: "The repo ships a terminal comparison (base model versus distilled, same weights, adapter toggled) and a browser playground. Both validate every output against the oracle and keep a running scoreboard.",
    code: `PYTHONPATH=.:src .venv/bin/python examples/tui_demo.py
PYTHONPATH=.:src .venv/bin/uvicorn examples.compare_demo:app --port 8010`,
  },
];

export default function DocsPage() {
  return (
    <>
      <SiteHeader ctaHref="/experiment" ctaLabel="See the experiment" />
      <main className="mx-auto max-w-[1200px] px-6 py-16 md:px-10">
        <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-black/50">
          Documentation
        </p>
        <h1 className="mt-3 font-serif text-[clamp(40px,5vw,72px)] leading-[1.02] tracking-[-0.03em]">
          Distillery docs
        </h1>
        <p className="mt-5 max-w-2xl font-serif text-xl leading-relaxed text-black/70">
          Everything you need to distill a large model into a small one you can
          prove: datasets, recipes, runs, and the gate that decides what ships.
        </p>

        <nav className="mt-10 flex flex-wrap gap-3">
          {sections.map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              className="border border-black/20 px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.14em] hover:border-black"
            >
              {s.kicker}
            </a>
          ))}
        </nav>

        <div className="mt-16 space-y-20">
          {sections.map((s) => (
            <section key={s.id} id={s.id} className="grid gap-8 md:grid-cols-2">
              <div>
                <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-[#d75a3d]">
                  {s.kicker}
                </p>
                <h2 className="mt-3 font-serif text-[34px] leading-[1.1] tracking-[-0.02em]">
                  {s.title}
                </h2>
                <p className="mt-4 max-w-md text-[15px] leading-7 text-black/70">
                  {s.body}
                </p>
              </div>
              <div className="border border-black/80 bg-[#fbfaf6]">
                <div className="flex items-center gap-2 border-b border-black/80 px-4 py-2.5">
                  <span className="size-2 rounded-full border border-black/80" />
                  <span className="size-2 rounded-full border border-black/80" />
                  <span className="size-2 rounded-full border border-black/80" />
                  <span className="ml-3 font-mono text-[12px]">{s.id}.py</span>
                </div>
                <pre className="overflow-x-auto p-5 font-mono text-[13px] leading-6">
                  {s.code}
                </pre>
              </div>
            </section>
          ))}
        </div>

        <div className="mt-24 border-t border-black/20 pt-10">
          <p className="font-serif text-2xl">
            Next:{" "}
            <Link href="/experiment" className="underline underline-offset-4">
              read the experiment
            </Link>{" "}
            or{" "}
            <Link href="/distillery" className="underline underline-offset-4">
              the product page
            </Link>
            .
          </p>
        </div>
      </main>
      <SiteFooter />
    </>
  );
}
