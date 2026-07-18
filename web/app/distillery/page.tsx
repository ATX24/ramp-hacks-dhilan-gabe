import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, ArrowUpRight, BookOpen, Check } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { SiteFooter, SiteHeader } from "../components/SiteChrome";

const DOCS_URL =
  "https://github.com/ATX24/ramp-hacks-dhilan-gabe/blob/main/project1/README.md";

export const metadata: Metadata = {
  title: "Distillery | Anthropic 2",
  description:
    "Distillery turns traces and large-model outputs into smaller portable models through inspectable, user-defined recipes.",
};

const stages = [
  {
    number: "01",
    title: "Curate",
    copy: "Import traces, isolate held-out splits, and freeze every source, label, and hash.",
  },
  {
    number: "02",
    title: "Synthesize",
    copy: "Ask a teacher only for missing training labels. Preserve provenance and reject invalid outputs.",
  },
  {
    number: "03",
    title: "Train",
    copy: "Resolve a built-in or user-defined recipe, then run one finite, sealed training job.",
  },
  {
    number: "04",
    title: "Evaluate",
    copy: "Compare the student on frozen data and make quality, cost, and failure states visible.",
  },
] as const;

function Quickstart() {
  return (
    <div className="min-w-0 overflow-hidden rounded-[20px] bg-[#141413] text-[#faf9f5] shadow-2xl shadow-black/20">
      <div className="flex items-center justify-between border-b border-white/15 px-5 py-4 font-mono text-[10px] tracking-[0.08em] text-white/55">
        <span>distill.py</span>
        <span className="text-[#d8ff8f]">● ready</span>
      </div>
      <pre className="max-w-full overflow-x-auto p-6 font-mono text-[12px] leading-7 md:p-8 md:text-[14px]">
        <code>{`distillery = Distillery(base_url=os.environ["DISTILLERY_URL"])
dataset = distillery.datasets.create("./finance_world.jsonl")
run = distillery.distill(dataset, recipe="auto").wait()`}</code>
      </pre>
      <div className="grid grid-cols-3 gap-px bg-white/15 font-mono text-[9px] uppercase tracking-[0.1em] text-white/50">
        <span className="bg-[#1d1d1b] p-4 text-center">immutable data</span>
        <span className="bg-[#1d1d1b] p-4 text-center">resolved recipe</span>
        <span className="bg-[#1d1d1b] p-4 text-center">portable model</span>
      </div>
    </div>
  );
}

export default function DistilleryPage() {
  return (
    <>
      <SiteHeader ctaHref={DOCS_URL} ctaLabel="View docs" />
      <main>
        <section className="mx-auto max-w-[1600px] px-6 pb-16 pt-8 md:px-10 lg:px-14">
          <div className="overflow-hidden rounded-[28px] bg-[#d65f45] p-7 text-[#141413] md:p-12 lg:p-16">
            <div className="flex justify-between font-mono text-[10px] uppercase tracking-[0.14em]">
              <span>Distillation platform</span>
              <span>Open research release · 2026</span>
            </div>

            <div className="mt-20 grid min-w-0 gap-14 md:grid-cols-[minmax(0,0.82fr)_minmax(0,1.18fr)] md:items-center lg:mt-28">
              <div className="flex min-h-[430px] flex-col justify-between">
                <div>
                  <h1 className="text-[clamp(72px,9vw,148px)] font-semibold leading-[0.8] tracking-[-0.075em]">
                    Distillery
                  </h1>
                  <p className="mt-10 max-w-xl font-serif text-[clamp(30px,3.2vw,50px)] leading-[1.08] tracking-[-0.025em]">
                    Describe how a large model becomes a small one. Keep the model—and the evidence.
                  </p>
                </div>
                <div className="mt-10 flex flex-wrap gap-3">
                  <Button
                    asChild
                    size="lg"
                    className="h-12 rounded-xl bg-[#141413] px-5 !text-[#faf9f5] hover:bg-black/80 hover:!text-white"
                  >
                    <a href={DOCS_URL} target="_blank" rel="noreferrer">
                      <BookOpen className="size-4" /> Read the docs <ArrowUpRight className="size-4" />
                    </a>
                  </Button>
                  <Button asChild size="lg" variant="outline" className="h-12 rounded-xl border-black/35 bg-transparent px-5 hover:bg-black/10">
                    <Link href="/experiment">See the experiment <ArrowRight className="size-4" /></Link>
                  </Button>
                </div>
              </div>
              <div className="min-w-0">
                <Quickstart />
              </div>
            </div>
          </div>
        </section>

        <section className="mx-auto grid max-w-[1600px] gap-16 px-6 py-24 md:grid-cols-[0.8fr_1.2fr] md:px-10 lg:px-14">
          <div>
            <Badge variant="outline" className="rounded-full border-black/25 bg-transparent font-mono text-[9px] uppercase tracking-[0.12em]">
              Recipes, not switches
            </Badge>
            <h2 className="mt-8 max-w-xl font-serif text-[clamp(42px,5vw,76px)] leading-[1.02] tracking-[-0.045em]">
              Your method is plain Python.
            </h2>
          </div>
          <div className="grid content-start gap-8 md:pt-2">
            <p className="max-w-3xl text-lg leading-8 text-black/65">
              Distillery exposes the actual decisions behind distillation: which labels to keep, when to call a teacher, what counts as valid, and which examples reach training. Built-in and user-defined recipes run through the same immutable data path and evaluation gate.
            </p>
            <Card className="rounded-[20px] border-0 bg-[#e8e4da] py-0 shadow-none ring-0">
              <CardContent className="p-0">
                <div className="border-b border-black/15 px-6 py-4 font-mono text-[10px] tracking-[0.1em] text-black/50">my_recipe.py</div>
                <pre className="overflow-x-auto p-6 font-mono text-[12px] leading-7 md:p-8 md:text-[14px]">
                  <code>{`class RejectHard:
    name = "reject_hard.v1"

    def run(self, ctx, examples):
        ctx.teacher_label(examples)
        return [ex for ex in examples
                if ctx.is_valid(ex) and ctx.agrees_with_oracle(ex)]`}</code>
                </pre>
              </CardContent>
            </Card>
          </div>
        </section>

        <section className="mx-auto max-w-[1600px] border-y border-black/20">
          <div className="grid md:grid-cols-4">
            {stages.map((stage) => (
              <article
                key={stage.number}
                className="flex min-h-[390px] flex-col justify-between border-b border-black/20 p-7 last:border-b-0 md:border-b-0 md:border-r md:last:border-r-0 lg:p-9"
              >
                <span className="font-mono text-[10px] tracking-[0.14em]">{stage.number}</span>
                <div>
                  <h2 className="font-serif text-[38px] leading-none tracking-[-0.035em]">{stage.title}</h2>
                  <p className="mt-6 text-[15px] leading-6 text-black/60">{stage.copy}</p>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="mx-auto max-w-[1600px] px-6 py-28 md:px-10 lg:px-14">
          <div className="grid overflow-hidden rounded-[26px] bg-[#141413] text-[#faf9f5] md:grid-cols-[1fr_0.9fr]">
            <div className="flex min-h-[560px] flex-col justify-between p-8 md:p-12 lg:p-16">
              <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-white/50">The first experiment</span>
              <div>
                <h2 className="max-w-3xl font-serif text-[clamp(42px,5vw,76px)] leading-[1.02] tracking-[-0.04em]">TinyFable is the test, not the testimonial.</h2>
                <p className="mt-8 max-w-2xl text-lg leading-8 text-white/60">
                  We ran two recipes through the same sealed pipeline: standard sequence distillation and a user-defined rejection-sampling method. The same 0.5B student, the same frozen evaluation, and losing states left intact.
                </p>
              </div>
              <Link href="/experiment" className="flex w-fit items-center gap-2 text-sm underline underline-offset-8">
                Read the experiment <ArrowRight className="size-4" />
              </Link>
            </div>
            <div className="grid place-items-center bg-[#d65f45] p-8 text-[#141413] md:p-12">
              <div className="grid w-full max-w-sm gap-4">
                {["Immutable datasets", "User-defined recipes", "Frozen evaluation", "Portable artifacts"].map((item) => (
                  <div key={item} className="flex items-center gap-4 rounded-xl border border-black/25 p-5 font-mono text-[11px] uppercase tracking-[0.1em]">
                    <Check className="size-4 shrink-0" /> {item}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="mx-auto max-w-[1600px] px-6 pb-8 md:px-10 lg:px-14">
          <div className="flex flex-col gap-10 border-t border-black/20 py-20 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="font-mono text-[10px] uppercase tracking-[0.14em]">Open documentation</p>
              <h2 className="mt-6 max-w-4xl text-[clamp(46px,6vw,92px)] font-semibold leading-[0.94] tracking-[-0.06em]">Clone it. Change the recipe. Run it yourself.</h2>
            </div>
            <Button asChild size="lg" className="h-12 shrink-0 rounded-xl bg-[#141413] px-5 !text-[#faf9f5] hover:bg-black/80 hover:!text-white">
              <a href={DOCS_URL} target="_blank" rel="noreferrer"><BookOpen className="size-4" /> View on GitHub <ArrowUpRight className="size-4" /></a>
            </Button>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
