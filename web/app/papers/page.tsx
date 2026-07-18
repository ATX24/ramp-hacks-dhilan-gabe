import type { Metadata } from "next";
import { ArrowUpRight, FileText } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { SiteFooter, SiteHeader } from "../components/SiteChrome";

export const metadata: Metadata = {
  title: "Papers | Anthropic 2",
  description: "Working papers from Anthropic 2 on TinyFable, Distillery, and model evaluation.",
};

const papers = [
  {
    number: "01",
    category: "Model report",
    title: "TinyFable: Distilling a Finance Generalist",
    abstract: "A portable 0.5B-class model trained to perform transaction review and variance analysis with the same weights, evaluated against rules, its base, its teacher, and practical alternatives.",
    href: "/papers/tinyfable-systems.pdf",
  },
  {
    number: "02",
    category: "Systems",
    title: "Distillery: Transparent Method Selection for Smaller Models",
    abstract: "A high-level system for deciding whether existing traces and synthetic examples should become a smaller portable model before billable training begins.",
    href: "/papers/durable-promises.pdf",
  },
  {
    number: "03",
    category: "Evaluation",
    title: "Evaluating TinyFable: Quality, Systems, and Economics",
    abstract: "A release evaluation that freezes identities, matches trainable arms, measures quality and systems behavior, and applies a utilization-sensitive economic gate.",
    href: "/papers/evaluation-economics.pdf",
  },
] as const;

export default function PapersPage() {
  return (
    <>
      <SiteHeader />
      <main>
        <section className="mx-auto grid max-w-[1600px] gap-16 px-6 pb-24 pt-20 md:grid-cols-[0.85fr_1.15fr] md:px-10 lg:px-14">
          <h1 className="text-[clamp(56px,6vw,92px)] font-semibold leading-none tracking-[-0.06em]">Papers</h1>
          <div className="max-w-3xl md:pt-3">
            <p className="font-serif text-[clamp(27px,2.6vw,42px)] leading-[1.2] tracking-[-0.025em]">We publish the model, the system, and the evaluation together.</p>
            <p className="mt-6 max-w-2xl text-base leading-7 text-black/60">Working papers formatted in the NeurIPS 2025 style. Authors: Gabriel Keller and Dhilan Shah, Anthropic 2.</p>
          </div>
        </section>

        <Separator className="bg-black/20" />

        <section className="mx-auto max-w-[1600px] px-6 md:px-10 lg:px-14">
          {papers.map((paper) => (
            <article key={paper.number} className="grid gap-8 border-b border-black/20 py-12 md:grid-cols-[90px_1.1fr_0.9fr_auto] md:py-16">
              <span className="font-mono text-[11px] tracking-[0.12em]">{paper.number}</span>
              <div>
                <Badge variant="outline" className="rounded-full border-black/25 bg-transparent font-mono text-[9px] uppercase tracking-[0.12em]">{paper.category}</Badge>
                <h2 className="mt-5 max-w-2xl font-serif text-[clamp(30px,3vw,48px)] leading-[1.08] tracking-[-0.03em]">{paper.title}</h2>
                <p className="mt-5 text-sm text-black/55">Gabriel Keller · Dhilan Shah · Anthropic 2 · 2026</p>
              </div>
              <p className="max-w-xl text-base leading-7 text-black/65">{paper.abstract}</p>
              <Button asChild variant="outline" className="h-11 w-fit rounded-xl border-black/30 bg-transparent px-4">
                <a href={paper.href} target="_blank" rel="noreferrer"><FileText className="size-4" /> PDF <ArrowUpRight className="size-4" /></a>
              </Button>
            </article>
          ))}
        </section>

        <section className="mx-auto grid max-w-[1600px] gap-12 px-6 py-28 md:grid-cols-2 md:px-10 lg:px-14">
          <h2 className="font-serif text-[clamp(38px,4.5vw,68px)] leading-[1.06] tracking-[-0.04em]">One claim.<br />Three layers of evidence.</h2>
          <p className="max-w-2xl text-lg leading-8 text-black/65">TinyFable is the model result. Distillery is the system that produced it. The evaluation paper explains why the release gate passed—and what the experiment still does not establish.</p>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
