import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, ArrowUpRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { SiteFooter, SiteHeader } from "../components/SiteChrome";

export const metadata: Metadata = {
  title: "Research | Anthropic 2",
  description: "Research at Anthropic 2 on compact models, distillation systems, and evidence-led evaluation.",
};

const areas = [
  {
    index: "01",
    title: "Compact models",
    copy: "We study when one smaller model can preserve useful behavior across multiple tasks without hiding specialists behind a router.",
  },
  {
    index: "02",
    title: "Distillation systems",
    copy: "We make method selection, provenance, manifests, and portable artifacts explicit enough to inspect and reproduce.",
  },
  {
    index: "03",
    title: "Model evaluation",
    copy: "We combine quality, uncertainty, out-of-distribution behavior, systems measurements, and deployment economics in one release decision.",
  },
  {
    index: "04",
    title: "Synthetic environments",
    copy: "We generate latent finance worlds before rendering prompts so accounting invariants and held-out regimes remain machine-checkable.",
  },
] as const;

export default function ResearchPage() {
  return (
    <>
      <SiteHeader />
      <main>
        <section className="mx-auto grid max-w-[1600px] gap-16 px-6 pb-24 pt-20 md:grid-cols-[0.85fr_1.15fr] md:px-10 lg:px-14">
          <h1 className="text-[clamp(56px,6vw,92px)] font-semibold leading-none tracking-[-0.06em]">Research</h1>
          <div className="max-w-3xl md:pt-3">
            <p className="font-serif text-[clamp(27px,2.6vw,42px)] leading-[1.2] tracking-[-0.025em]">
              We study when smaller models are genuinely better deployment choices—and build the systems needed to find out.
            </p>
            <div className="mt-8 flex flex-wrap gap-x-7 gap-y-3 text-sm underline underline-offset-8">
              <a href="#compact-models">Compact models</a>
              <a href="#distillation-systems">Distillation systems</a>
              <a href="#evaluation">Evaluation</a>
              <a href="#synthetic-environments">Synthetic environments</a>
            </div>
          </div>
        </section>

        <section className="mx-auto grid max-w-[1600px] border-y border-black/20 md:grid-cols-4">
          {areas.map((area, index) => (
            <article
              key={area.title}
              id={["compact-models", "distillation-systems", "evaluation", "synthetic-environments"][index]}
              className="flex min-h-[420px] flex-col justify-between border-b border-black/20 p-7 last:border-b-0 md:border-b-0 md:border-r md:last:border-r-0 lg:p-9"
            >
              <span className="font-mono text-[10px] tracking-[0.14em]">{area.index}</span>
              <div>
                <h2 className="font-serif text-[36px] leading-none tracking-[-0.035em]">{area.title}</h2>
                <p className="mt-6 text-[15px] leading-6 text-black/65">{area.copy}</p>
              </div>
            </article>
          ))}
        </section>

        <section className="mx-auto max-w-[1600px] px-6 py-28 md:px-10 lg:px-14">
          <div className="mb-10 flex items-end justify-between">
            <div>
              <Badge variant="outline" className="rounded-full border-black/25 bg-transparent font-mono text-[10px] uppercase tracking-[0.12em]">Featured work</Badge>
              <h2 className="mt-6 text-[clamp(40px,4.8vw,72px)] font-semibold leading-none tracking-[-0.055em]">TinyFable</h2>
            </div>
            <span className="font-mono text-[10px] uppercase tracking-[0.12em]">Model 001 · July 2026</span>
          </div>
          <Card className="grid min-h-[520px] overflow-hidden rounded-[26px] border-0 bg-[#141413] py-0 text-[#faf9f5] shadow-none ring-0 md:grid-cols-[1.05fr_0.95fr]">
            <CardContent className="flex flex-col justify-between p-8 md:p-12 lg:p-16">
              <p className="max-w-3xl font-serif text-[clamp(34px,4.4vw,68px)] leading-[1.04] tracking-[-0.035em]">
                Can a 0.5B model handle two finance tasks with one set of weights—and justify the cost of making it?
              </p>
              <Link href="/tinyfable" className="mt-12 flex w-fit items-center gap-2 text-sm underline underline-offset-8">
                Explore TinyFable <ArrowUpRight className="size-4" />
              </Link>
            </CardContent>
            <div className="grid place-items-center bg-[#d65f45] p-10 text-[#141413]">
              <div className="grid w-full max-w-sm gap-4">
                <div className="rounded-2xl border border-black/25 p-6">
                  <span className="font-mono text-[10px] uppercase tracking-[0.12em]">Teacher</span>
                  <strong className="mt-3 block font-serif text-4xl font-normal">1.5B</strong>
                </div>
                <ArrowRight className="mx-auto size-6 rotate-90" />
                <div className="rounded-2xl bg-[#141413] p-6 text-[#faf9f5]">
                  <span className="font-mono text-[10px] uppercase tracking-[0.12em]">TinyFable</span>
                  <strong className="mt-3 block font-serif text-4xl font-normal">0.5B</strong>
                </div>
              </div>
            </div>
          </Card>
        </section>

        <section className="mx-auto grid max-w-[1600px] gap-12 border-t border-black/20 px-6 py-24 md:grid-cols-2 md:px-10 lg:px-14">
          <h2 className="font-serif text-[clamp(36px,4vw,62px)] leading-[1.08] tracking-[-0.035em]">Our default result can be: do not distill.</h2>
          <div>
            <p className="max-w-2xl text-lg leading-8 text-black/65">
              A completed training job is not evidence that a model should ship. Our experiments preserve losing arms, record the first failed gate, and keep quality claims bounded to frozen data and declared assumptions.
            </p>
            <Link href="/papers" className="mt-8 flex w-fit items-center gap-2 text-sm underline underline-offset-8">Read the papers <ArrowRight className="size-4" /></Link>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
