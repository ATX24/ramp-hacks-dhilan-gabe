import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, Mail } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { SiteFooter, SiteHeader } from "../components/SiteChrome";

export const metadata: Metadata = {
  title: "Company | Anthropic 2",
  description: "About Anthropic 2, an independent hackathon research lab founded by Gabriel Keller and Dhilan Shah.",
};

export default function CompanyPage() {
  return (
    <>
      <SiteHeader />
      <main>
        <section className="mx-auto max-w-[1600px] px-6 pb-28 pt-20 md:px-10 lg:px-14">
          <p className="font-mono text-[10px] uppercase tracking-[0.14em]">About Anthropic 2</p>
          <h1 className="mt-12 max-w-[1250px] text-[clamp(54px,8.2vw,132px)] font-semibold leading-[0.92] tracking-[-0.07em]">
            We build smaller models—and make them prove they deserve to exist.
          </h1>
        </section>

        <section className="bg-[#d65f45] text-[#141413]">
          <div className="mx-auto grid min-h-[650px] max-w-[1600px] gap-16 px-6 py-24 md:grid-cols-2 md:px-10 lg:px-14">
            <h2 className="font-serif text-[clamp(40px,5vw,78px)] leading-[1.03] tracking-[-0.04em]">An independent frontier research lab, built in one very concentrated hackathon.</h2>
            <div className="grid content-end gap-7 text-lg leading-8">
              <p>Anthropic 2 studies a practical question: when should capability that lives in traces and large models become a smaller portable model?</p>
              <p>Our first answer is TinyFable, a 0.5B finance generalist. Our second is Distillery, the system that curates the data, selects a method, trains the candidate, and decides whether the result is actually worth deploying.</p>
              <p>We publish the model claim, the system, and the evaluation separately so each can be challenged on its own terms.</p>
            </div>
          </div>
        </section>

        <section className="mx-auto max-w-[1600px] px-6 py-28 md:px-10 lg:px-14">
          <h2 className="mb-10 text-[clamp(40px,4vw,62px)] font-semibold tracking-[-0.05em]">Founders</h2>
          <div className="grid gap-4 md:grid-cols-2">
            {[
              { name: "Gabriel Keller", email: "gabrielkeller@utexas.edu" },
              { name: "Dhilan Shah", email: "dhilan.shah@utexas.edu" },
            ].map((person, index) => (
              <Card key={person.name} className="min-h-[330px] justify-between rounded-[22px] border-0 bg-[#e8e4da] py-0 shadow-none ring-0">
                <CardContent className="flex h-full flex-col justify-between p-8">
                  <span className="font-mono text-[10px] uppercase tracking-[0.14em]">Founder 0{index + 1}</span>
                  <div>
                    <h3 className="font-serif text-[clamp(40px,4vw,62px)] leading-none tracking-[-0.04em]">{person.name}</h3>
                    <a className="mt-6 flex w-fit items-center gap-2 text-sm underline underline-offset-8" href={`mailto:${person.email}`}><Mail className="size-4" /> {person.email}</a>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        <section className="mx-auto grid max-w-[1600px] gap-12 border-t border-black/20 px-6 py-24 md:grid-cols-2 md:px-10 lg:px-14">
          <h2 className="font-serif text-[clamp(36px,4vw,62px)] leading-[1.08] tracking-[-0.035em]">Yes, the name is satire.</h2>
          <div>
            <p className="max-w-2xl text-lg leading-8 text-black/65">Anthropic 2 is not affiliated with Anthropic. The visual resemblance is deliberate; the extra square “2” is doing the legal and comedic heavy lifting. The research, code, papers, and questionable sleep schedule are ours.</p>
            <Button asChild variant="link" className="mt-6 px-0 text-[#141413] underline underline-offset-8"><Link href="/research">See our research <ArrowRight /></Link></Button>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
