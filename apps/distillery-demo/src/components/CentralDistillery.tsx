"use client";

import { FlaskConical, GraduationCap } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { StageRouteContent } from "@/components/StageRouteContent";
import { Button } from "@/components/ui/button";
import type { StageBundle } from "@/lib/types";

export type CentralStage = "train" | "demo";

export function CentralDistillery({
  bundle,
  initialStage,
}: {
  bundle: StageBundle;
  initialStage: CentralStage;
}) {
  const [stage, setStage] = useState<CentralStage>(initialStage);
  const router = useRouter();
  const searchParams = useSearchParams();

  function chooseStage(nextStage: CentralStage) {
    setStage(nextStage);
    const params = new URLSearchParams(searchParams.toString());
    params.set("stage", nextStage);
    router.replace(`/?${params.toString()}`, { scroll: false });
  }

  return (
    <div className="grid gap-4">
      <nav
        className="flex flex-col gap-4 rounded-[20px] bg-[#141413] p-4 text-[#faf9f5] sm:flex-row sm:items-center sm:justify-between sm:p-5"
        aria-label="Distillery workflow"
        data-testid="central-stage-control"
      >
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-white/50">
            Distillery workflow
          </p>
          <p className="mt-1 max-w-lg font-serif text-lg text-white/80">
            Teach with an example, then try both model versions.
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            type="button"
            size="lg"
            variant={stage === "train" ? "default" : "ghost"}
            className={
              stage === "train"
                ? "bg-[#d65f45] text-[#141413] hover:bg-[#e96242]"
                : "text-white hover:bg-white/10 hover:text-white"
            }
            aria-pressed={stage === "train"}
            data-testid="central-stage-train"
            onClick={() => chooseStage("train")}
          >
            <GraduationCap aria-hidden />
            Teach model
          </Button>
          <Button
            type="button"
            size="lg"
            variant={stage === "demo" ? "default" : "ghost"}
            className={
              stage === "demo"
                ? "bg-[#d65f45] text-[#141413] hover:bg-[#e96242]"
                : "text-white hover:bg-white/10 hover:text-white"
            }
            aria-pressed={stage === "demo"}
            data-testid="central-stage-demo"
            onClick={() => chooseStage("demo")}
          >
            <FlaskConical aria-hidden />
            Try and compare
          </Button>
        </div>
      </nav>

      <StageRouteContent stage={stage} bundle={bundle} />
    </div>
  );
}
