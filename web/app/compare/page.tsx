import type { Metadata } from "next";
import { SiteFooter, SiteHeader } from "../components/SiteChrome";
import { ModelCompare } from "./ModelCompare";

export const metadata: Metadata = {
  title: "Compare | Distillery",
  description:
    "Run the base 0.5B model and the distilled student side by side on held-out finance prompts, with every output oracle-checked.",
};

export default function ComparePage() {
  return (
    <>
      <SiteHeader ctaHref="/docs" ctaLabel="Read the docs" />
      <main className="mx-auto max-w-[1300px] px-6 py-16 md:px-10">
        <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-black/50">
          Model comparison
        </p>
        <h1 className="mt-3 font-serif text-[clamp(40px,5vw,72px)] leading-[1.02] tracking-[-0.03em]">
          Base vs distilled
        </h1>
        <p className="mt-5 max-w-2xl font-serif text-xl leading-relaxed text-black/70">
          The same 0.5B weights answer every prompt twice: once with the LoRA
          adapter off (base) and once with it on (distilled from a frontier
          teacher). Prompts are frozen held-out finance cases neither model
          trained on, and every output is validated against the executable
          oracle. Outputs are real captures from the local evaluation harness.
        </p>
        <ModelCompare />
      </main>
      <SiteFooter />
    </>
  );
}
