"use client";

import Link from "next/link";
import { ArrowRight, ArrowUpRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/ui/carousel";

function CodePanel() {
  return (
    <div className="overflow-hidden rounded-2xl bg-[#141413] text-[#faf9f5] shadow-2xl shadow-black/20">
      <div className="flex items-center justify-between border-b border-white/15 px-5 py-3 font-mono text-[11px] text-white/55">
        <span>distill.py</span>
        <span className="text-[#d8ff8f]">● ready</span>
      </div>
      <pre className="overflow-x-auto p-5 font-mono text-[12px] leading-7 md:p-7 md:text-[14px]">
        <code>{`distillery = Distillery(api_key=os.environ["DISTILLERY_API_KEY"])
dataset = distillery.datasets.create("./finance_world.jsonl")
run = distillery.distill(dataset, recipe="auto").wait()`}</code>
      </pre>
    </div>
  );
}

export function ProductStage() {
  return (
    <Carousel opts={{ loop: true, startIndex: 1 }} className="mx-auto w-full max-w-[1600px] px-6 md:px-10 lg:px-14">
      <div className="relative overflow-hidden rounded-[28px] bg-[#141413]">
        <CarouselContent className="ml-0">
          <CarouselItem className="pl-0">
            <article className="grid min-h-[620px] gap-10 p-6 text-[#141413] md:grid-cols-[0.9fr_1.1fr] md:p-10 lg:p-14">
              <div className="flex flex-col justify-between rounded-[20px] bg-[#f1eee6] p-7 md:p-10">
                <div className="flex justify-between font-mono text-[10px] tracking-[0.12em]">
                  <span>MODEL 001</span>
                  <span>JULY 2026</span>
                </div>
                <div>
                  <p className="mb-5 font-mono text-[10px] uppercase tracking-[0.14em]">Trained with Distillery</p>
                  <h2 className="font-serif text-[clamp(72px,9vw,150px)] font-normal leading-[0.78] tracking-[-0.07em]">TinyFable</h2>
                </div>
                <Button asChild variant="link" className="w-fit px-0 text-[#141413] underline underline-offset-8">
                  <Link href="/tinyfable">Read the model report <ArrowRight /></Link>
                </Button>
              </div>
              <div className="flex flex-col justify-between p-2 text-[#faf9f5] md:p-5">
                <p className="max-w-xl font-serif text-[clamp(34px,4vw,62px)] leading-[1.02] tracking-[-0.03em]">
                  One portable finance generalist. Distilled from a 1.5B teacher into a 0.5B model.
                </p>
                <div className="grid grid-cols-3 gap-px overflow-hidden rounded-2xl bg-white/15">
                  {[["0.5B", "parameters"], ["2", "finance tasks"], ["1", "set of weights"]].map(([value, label]) => (
                    <div key={label} className="bg-[#1d1d1b] p-5">
                      <strong className="block font-serif text-3xl font-normal">{value}</strong>
                      <span className="text-xs text-white/55">{label}</span>
                    </div>
                  ))}
                </div>
              </div>
            </article>
          </CarouselItem>

          <CarouselItem className="pl-0">
            <article className="grid min-h-[620px] gap-10 bg-[#d65f45] p-8 text-[#141413] md:grid-cols-[0.85fr_1.15fr] md:p-14 lg:p-20">
              <div className="flex flex-col justify-between">
                <div className="font-mono text-[10px] uppercase tracking-[0.13em]">Distillation product · available now</div>
                <div>
                  <h2 className="text-[clamp(64px,8vw,132px)] font-semibold leading-[0.82] tracking-[-0.07em]">Distillery</h2>
                  <p className="mt-8 max-w-lg font-serif text-3xl leading-tight">From a dataset to a portable model in three lines.</p>
                </div>
                <Button asChild className="w-fit rounded-xl bg-[#141413] text-[#faf9f5] hover:bg-black/80">
                  <Link href="/distillery">Open Distillery <ArrowUpRight /></Link>
                </Button>
              </div>
              <div className="flex items-center"><CodePanel /></div>
            </article>
          </CarouselItem>
        </CarouselContent>
        <div className="absolute bottom-5 right-5 flex gap-2">
          <CarouselPrevious className="static translate-y-0 border-white/20 bg-[#141413]/80 text-white hover:bg-white hover:text-black" />
          <CarouselNext className="static translate-y-0 border-white/20 bg-[#141413]/80 text-white hover:bg-white hover:text-black" />
        </div>
      </div>
    </Carousel>
  );
}
