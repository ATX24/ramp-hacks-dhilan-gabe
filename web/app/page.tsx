import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";
import { ProductStage } from "./components/ProductStage";
import { SiteFooter, SiteHeader } from "./components/SiteChrome";

export const metadata: Metadata = {
  title: "Anthropic 2 | AI research and products",
  description: "Anthropic 2 builds smaller models and the systems that make them worth deploying.",
};

const releases = [
  {
    eyebrow: "Model release",
    title: "Introducing TinyFable",
    copy: "A 0.5B finance generalist trained with Distillery and evaluated against its teacher.",
    href: "/tinyfable",
    date: "July 2026",
  },
  {
    eyebrow: "Product",
    title: "Distillery",
    copy: "Turn an evaluation dataset into a smaller, portable model with one resolved recipe.",
    href: "/distillery",
    date: "July 2026",
  },
  {
    eyebrow: "Research",
    title: "When should a model be smaller?",
    copy: "A systems-and-economics evaluation of distillation as a deployment decision.",
    href: "/papers",
    date: "July 2026",
  },
] as const;

export default function Home() {
  return (
    <>
      <SiteHeader />
      <main>
        <section className="mx-auto grid min-h-[530px] max-w-[1600px] items-center gap-16 px-6 py-20 md:grid-cols-2 md:px-10 lg:px-14">
          <h1 className="max-w-3xl text-[clamp(48px,5.7vw,90px)] font-semibold leading-[1.02] tracking-[-0.055em]">
            AI <Link className="underline decoration-2 underline-offset-[10px]" href="/research">research</Link> and <Link className="underline decoration-2 underline-offset-[10px]" href="/distillery">products</Link> that make smaller models worth building
          </h1>
          <p className="max-w-2xl font-serif text-[clamp(24px,2.15vw,35px)] leading-[1.32] tracking-[-0.015em] md:justify-self-end">
            Anthropic 2 is an independent research lab building compact models and the infrastructure to decide—honestly—when they beat the alternatives.
          </p>
        </section>

        <ProductStage />

        <section className="mx-auto max-w-[1600px] px-6 py-28 md:px-10 lg:px-14">
          <div className="mb-10 flex items-end justify-between">
            <h2 className="text-[clamp(38px,4vw,62px)] font-semibold tracking-[-0.045em]">Latest releases</h2>
            <Link href="/papers" className="hidden items-center gap-2 text-sm underline underline-offset-8 md:flex">View all research <ArrowRight className="size-4" /></Link>
          </div>
          <div className="grid gap-4 md:grid-cols-3">
            {releases.map((release) => (
              <Card key={release.title} className="min-h-[390px] justify-between rounded-[22px] border-0 bg-[#e8e4da] py-0 shadow-none ring-0 transition-transform duration-300 hover:-translate-y-1">
                <CardHeader className="p-7">
                  <div className="flex justify-between font-mono text-[10px] uppercase tracking-[0.12em] text-black/55">
                    <span>{release.eyebrow}</span><span>{release.date}</span>
                  </div>
                </CardHeader>
                <CardContent className="p-7">
                  <h3 className="font-serif text-[38px] leading-[1.05] tracking-[-0.03em]">{release.title}</h3>
                  <p className="mt-5 max-w-sm text-base leading-6 text-black/65">{release.copy}</p>
                </CardContent>
                <CardFooter className="border-0 bg-transparent p-7">
                  <Link href={release.href} className="flex items-center gap-2 text-sm underline underline-offset-8">Read more <ArrowRight className="size-4" /></Link>
                </CardFooter>
              </Card>
            ))}
          </div>
        </section>

        <section className="mx-auto grid max-w-[1600px] gap-16 border-t border-black/20 px-6 py-28 md:grid-cols-2 md:px-10 lg:px-14">
          <h2 className="max-w-2xl font-serif text-[clamp(38px,4vw,64px)] leading-[1.08] tracking-[-0.035em]">We build models that earn the right to be smaller.</h2>
          <div className="divide-y divide-black/20 border-t border-black/20">
            {[['Models', 'TinyFable is our first portable finance generalist.', '/tinyfable'], ['Systems', 'Distillery makes method selection and evaluation reproducible.', '/distillery'], ['Evaluation', 'Quality, uncertainty, systems behavior, and economics share one release gate.', '/research']].map(([label, copy, href]) => (
              <Link href={href} key={label} className="grid grid-cols-[120px_1fr_auto] gap-4 py-6 hover:bg-black/[0.025]">
                <span className="font-mono text-[11px] uppercase tracking-[0.12em]">{label}</span>
                <span>{copy}</span>
                <ArrowRight className="size-4" />
              </Link>
            ))}
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
