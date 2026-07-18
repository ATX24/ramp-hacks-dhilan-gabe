import type { Metadata } from "next";
import Link from "next/link";
import { ArrowDown, ArrowRight, ArrowUpRight, Check } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { SiteFooter, SiteHeader } from "../components/SiteChrome";

export const metadata: Metadata = {
  title: "TinyFable | Anthropic 2",
  description: "TinyFable is a portable 0.5B finance-generalist model trained with Distillery.",
};

const gates = [
  ["Primary-task quality", "Passed"],
  ["OOD and uncertainty", "Passed"],
  ["Systems behavior", "Measured"],
  ["Economic gate", "Passed"],
] as const;

export default function TinyFablePage() {
  return (
    <>
      <SiteHeader />
      <main>
        <section className="mx-auto min-h-[calc(100svh-96px)] max-w-[1600px] px-6 pb-12 pt-14 md:px-10 lg:px-14">
          <div className="flex justify-between font-mono text-[10px] uppercase tracking-[0.14em]"><span>Model 001</span><span>Released July 2026</span></div>
          <h1 className="mt-[12vh] font-serif text-[clamp(88px,17vw,260px)] font-normal leading-[0.72] tracking-[-0.085em]">TinyFable</h1>
          <div className="mt-16 grid gap-10 md:grid-cols-[1fr_0.85fr] md:items-end">
            <div className="flex flex-wrap gap-3">
              <Badge variant="outline" className="rounded-full border-black/30 bg-transparent px-4 py-2 font-mono text-[10px] uppercase tracking-[0.11em]">0.5B parameters</Badge>
              <Badge variant="outline" className="rounded-full border-black/30 bg-transparent px-4 py-2 font-mono text-[10px] uppercase tracking-[0.11em]">2 primary tasks</Badge>
              <Badge variant="outline" className="rounded-full border-black/30 bg-transparent px-4 py-2 font-mono text-[10px] uppercase tracking-[0.11em]">Portable weights</Badge>
            </div>
            <div>
              <p className="font-serif text-[clamp(28px,3vw,46px)] leading-[1.14] tracking-[-0.025em]">Our first small model. Trained with Distillery. Evaluated against the alternatives.</p>
              <Button asChild variant="link" className="mt-6 px-0 text-[#141413] underline underline-offset-8"><a href="#model">Meet the model <ArrowDown /></a></Button>
            </div>
          </div>
        </section>

        <section id="model" className="bg-[#141413] text-[#faf9f5]">
          <div className="mx-auto grid min-h-[720px] max-w-[1600px] gap-16 px-6 py-20 md:grid-cols-[0.9fr_1.1fr] md:px-10 lg:px-14">
            <div className="flex flex-col justify-between">
              <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-white/55">One model · two finance jobs</span>
              <div>
                <h2 className="max-w-xl font-serif text-[clamp(46px,5.5vw,84px)] leading-[0.98] tracking-[-0.045em]">A finance generalist small enough to carry.</h2>
                <p className="mt-8 max-w-xl text-lg leading-8 text-white/65">TinyFable handles transaction review and variance analysis with the same weights. No hidden specialist router. No second checkpoint behind the demo.</p>
              </div>
              <div className="flex flex-wrap gap-6 text-sm underline underline-offset-8">
                <a href="/papers/tinyfable-systems.pdf" target="_blank" rel="noreferrer">Read the model report <ArrowUpRight className="inline size-4" /></a>
                <Link href="/distillery">See how it was trained <ArrowRight className="inline size-4" /></Link>
              </div>
            </div>
            <div className="grid place-items-center rounded-[24px] bg-[#d65f45] p-8 text-[#141413] md:p-14">
              <div className="w-full max-w-2xl">
                <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-5">
                  <div className="rounded-2xl border border-black/30 p-6 md:p-8"><span className="font-mono text-[9px] uppercase tracking-[0.12em]">Qwen2.5 teacher</span><strong className="mt-4 block font-serif text-5xl font-normal">1.5B</strong></div>
                  <ArrowRight className="size-6" />
                  <div className="rounded-2xl bg-[#141413] p-6 text-[#faf9f5] md:p-8"><span className="font-mono text-[9px] uppercase tracking-[0.12em]">TinyFable</span><strong className="mt-4 block font-serif text-5xl font-normal">0.5B</strong></div>
                </div>
                <div className="mt-5 grid grid-cols-2 gap-5 font-mono text-[9px] uppercase tracking-[0.1em]"><span className="rounded-full border border-black/30 px-4 py-3 text-center">sequence.v1</span><span className="rounded-full border border-black/30 px-4 py-3 text-center">logit.v1</span></div>
              </div>
            </div>
          </div>
        </section>

        <section className="mx-auto max-w-[1600px] px-6 py-28 md:px-10 lg:px-14">
          <div className="grid gap-16 md:grid-cols-[0.8fr_1.2fr]">
            <div>
              <p className="font-mono text-[10px] uppercase tracking-[0.14em]">Release evaluation</p>
              <h2 className="mt-7 font-serif text-[clamp(40px,4.7vw,72px)] leading-[1.04] tracking-[-0.04em]">Approved, within the experiment.</h2>
              <p className="mt-7 max-w-xl text-base leading-7 text-black/60">TinyFable cleared the frozen release gates. The claim remains bounded to the synthetic finance benchmark and declared economic assumptions.</p>
            </div>
            <Card className="rounded-[22px] border-0 bg-[#e8e4da] py-0 shadow-none ring-0">
              <CardContent className="divide-y divide-black/20 p-0">
                {gates.map(([label, result]) => (
                  <div key={label} className="grid grid-cols-[1fr_auto] items-center gap-5 p-6 md:p-8"><span className="text-lg">{label}</span><span className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.12em]"><Check className="size-4" /> {result}</span></div>
                ))}
              </CardContent>
            </Card>
          </div>
        </section>

        <section className="mx-auto max-w-[1600px] px-6 pb-12 md:px-10 lg:px-14">
          <div className="rounded-[26px] bg-[#d65f45] p-8 md:p-14 lg:p-20">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em]">Built with Distillery</p>
            <div className="mt-10 grid gap-12 md:grid-cols-[1fr_auto] md:items-end">
              <h2 className="max-w-4xl text-[clamp(48px,6vw,94px)] font-semibold leading-[0.93] tracking-[-0.06em]">Bring a dataset. Keep the model.</h2>
              <Button asChild className="h-12 rounded-xl bg-[#141413] px-5 !text-[#faf9f5] hover:bg-black/80"><Link href="/distillery">Open Distillery <ArrowUpRight /></Link></Button>
            </div>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
