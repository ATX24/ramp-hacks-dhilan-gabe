import { Suspense, type ReactNode } from "react";
import { StageStateBoundary } from "@/components/StageStateBoundary";
import { CurateStage } from "@/components/stages/CurateStage";
import { DemoStage } from "@/components/stages/DemoStage";
import { ProveStage } from "@/components/stages/ProveStage";
import { SynthesizeStage } from "@/components/stages/SynthesizeStage";
import { TrainStage } from "@/components/stages/TrainStage";
import { hasPrecomputedPriorCompletion } from "@/lib/runPresentation";
import type { StageBundle, StageId } from "@/lib/types";

const STAGE_NAMES: Record<StageId, string> = {
  curate: "Curate",
  synthesize: "Synthesize",
  train: "Train",
  prove: "Prove",
  demo: "Demo",
};

export function StageRouteContent({
  stage,
  bundle,
}: {
  stage: StageId;
  bundle: StageBundle;
}) {
  let content: ReactNode;

  switch (stage) {
    case "curate":
      content = (
        <CurateStage
          dataset={bundle.dataset}
          error={bundle.error}
          mode={bundle.mode}
          runId={bundle.run.run_id}
        />
      );
      break;

    case "synthesize":
      content = (
        <SynthesizeStage
          synthesis={bundle.synthesis}
          error={bundle.error}
          priorRun={hasPrecomputedPriorCompletion(bundle.run, bundle.artifact)}
        />
      );
      break;

    case "train":
      content = (
        <TrainStage
          mode={bundle.mode}
          plan={bundle.plan}
          run={bundle.run}
          artifact={bundle.artifact}
          telemetry={bundle.training_telemetry}
          error={bundle.error}
        />
      );
      break;

    case "prove":
      content = <ProveStage proof={bundle.proof} error={bundle.error} />;
      break;

    case "demo":
      content = (
        <Suspense
          fallback={
            <p data-testid="demo-suspense" role="status">
              Loading Demo / Playground…
            </p>
          }
        >
          <DemoStage bundle={bundle} />
        </Suspense>
      );
      break;

    default: {
      const _exhaustive: never = stage;
      content = _exhaustive;
    }
  }

  return (
    <StageStateBoundary state={bundle.load_state} stageName={STAGE_NAMES[stage]}>
      {content}
    </StageStateBoundary>
  );
}
