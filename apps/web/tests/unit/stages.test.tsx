import { cleanup, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CurateStage } from "@/components/stages/CurateStage";
import { ProveStage } from "@/components/stages/ProveStage";
import { SynthesizeStage } from "@/components/stages/SynthesizeStage";
import { TrainStage } from "@/components/stages/TrainStage";
import { StageNav, STAGE_ROUTES } from "@/components/StageNav";
import { buildStageBundle } from "@/lib/fixtures/bundle";
import { isRunCancellable } from "@/lib/runPresentation";

vi.mock("next/navigation", () => ({
  usePathname: () => "/curate",
}));

vi.mock("next/link", () => ({
  default: ({
    href,
    children,
    ...rest
  }: {
    href: string;
    children: ReactNode;
    [key: string]: unknown;
  }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

afterEach(() => {
  cleanup();
});

describe("stage components", () => {
  it("Curate shows mixture, hashes, and leakage", () => {
    const bundle = buildStageBundle("default");
    render(
      <CurateStage
        dataset={bundle.dataset}
        error={null}
        mode={bundle.mode}
        runId={bundle.run.run_id}
      />,
    );
    expect(screen.getByRole("heading", { name: "Curate" })).toBeInTheDocument();
    expect(screen.getByText("Task mixture")).toBeInTheDocument();
    expect(screen.getByText("Leakage checks")).toBeInTheDocument();
    expect(screen.getByText(/content:/i)).toBeInTheDocument();
  });

  it("Synthesize shows skipped state", () => {
    const bundle = buildStageBundle("skipped_synthesis");
    render(<SynthesizeStage synthesis={bundle.synthesis} error={null} />);
    expect(screen.getByTestId("synthesis-skipped")).toHaveTextContent(
      "responses_already_present",
    );
  });

  it("Train renders preparation-only state honestly", () => {
    const bundle = buildStageBundle("no_training_yet");
    render(
      <TrainStage
        mode={bundle.mode}
        plan={bundle.plan}
        run={bundle.run}
        artifact={bundle.artifact}
        telemetry={bundle.training_telemetry}
        error={null}
      />,
    );
    expect(screen.getByTestId("run-presentation")).toHaveAttribute(
      "data-presentation",
      "preparation",
    );
    expect(screen.getByTestId("job-activity")).toHaveTextContent("none");
    expect(screen.getByTestId("training-telemetry-not-started")).toHaveTextContent(
      "Not started · no metrics",
    );
  });

  it("Train exposes cancellation only for an active started run", () => {
    const bundle = buildStageBundle("default");
    expect(isRunCancellable(bundle.run, bundle.plan)).toBe(false);

    bundle.run.state = "TRAINING";
    bundle.run.training_launched = true;
    bundle.plan.planned_job.launched = true;
    expect(isRunCancellable(bundle.run, bundle.plan)).toBe(true);

    bundle.run.state = "SUCCEEDED";
    expect(isRunCancellable(bundle.run, bundle.plan)).toBe(false);
  });

  it("Curate CTA continues locally while preserving mode", () => {
    const bundle = buildStageBundle("insufficient_evidence");
    render(
      <CurateStage
        dataset={bundle.dataset}
        error={null}
        mode={bundle.mode}
        runId={bundle.run.run_id}
      />,
    );
    expect(screen.getByTestId("curate-continue")).toHaveAttribute(
      "href",
      "/synthesize?mode=insufficient_evidence&run=run_fixture_insufficient_evidence_001",
    );
  });

  it("Curate blocks continuation when the dataset is not frozen", () => {
    const bundle = buildStageBundle("error");
    render(
      <CurateStage
        dataset={bundle.dataset}
        error={bundle.error}
        mode={bundle.mode}
        runId={bundle.run.run_id}
      />,
    );
    expect(screen.getByTestId("curate-blocked")).toBeDisabled();
    expect(screen.queryByTestId("curate-continue")).not.toBeInTheDocument();
  });

  it("Prove shows projected economics and insufficient evidence", () => {
    const bundle = buildStageBundle("insufficient_evidence");
    render(<ProveStage proof={bundle.proof} error={null} />);
    expect(screen.getByText("insufficient_evidence")).toBeInTheDocument();
    expect(screen.getAllByText(/Projected/).length).toBeGreaterThan(0);
    expect(screen.getByText("Arm comparison")).toBeInTheDocument();
  });

  it("Train labels prior-run events and metrics as immutable", () => {
    const bundle = buildStageBundle("proved");
    render(
      <TrainStage
        mode={bundle.mode}
        plan={bundle.plan}
        run={bundle.run}
        artifact={bundle.artifact}
        telemetry={bundle.training_telemetry}
        error={null}
      />,
    );
    expect(screen.getByTestId("training-telemetry-prior")).toHaveTextContent(
      "Immutable prior-run record",
    );
    expect(screen.getAllByText("completion_ce")).toHaveLength(3);
    expect(screen.getByText("30")).toBeInTheDocument();
  });

  it("Train renders preparation telemetry errors safely", () => {
    const bundle = buildStageBundle("error");
    render(
      <TrainStage
        mode={bundle.mode}
        plan={bundle.plan}
        run={bundle.run}
        artifact={bundle.artifact}
        telemetry={bundle.training_telemetry}
        error={bundle.error}
      />,
    );
    expect(screen.getByTestId("training-telemetry-error")).toHaveTextContent(
      "Metrics unavailable",
    );
  });

  it("derives measured economics labels from the contract field", () => {
    const bundle = buildStageBundle("proved");
    const proof = structuredClone(bundle.proof);
    expect(proof).not.toBeNull();
    if (!proof) return;
    proof.economics.serving_cost_projected = false;
    proof.economics.note = "Serving costs are measured prior-run values.";
    proof.limitations = proof.limitations.map((limitation) =>
      limitation.replace(
        "Serving economics are projected at disclosed utilization points.",
        "Serving economics are measured prior-run values at disclosed utilization points.",
      ),
    );
    render(<ProveStage proof={proof} error={null} />);
    expect(screen.getByText("Measured serving economics")).toBeInTheDocument();
    expect(screen.getByText("Measured $/request")).toBeInTheDocument();
  });

  it("Prove empty state when no report", () => {
    render(<ProveStage proof={null} error={null} />);
    expect(screen.getByTestId("prove-empty")).toBeInTheDocument();
  });
});

describe("stage navigation contract", () => {
  it("exposes exactly four stage routes", () => {
    expect(STAGE_ROUTES).toEqual([
      "/curate",
      "/synthesize",
      "/train",
      "/prove",
    ]);
  });

  it("renders accessible nav links", () => {
    render(<StageNav mode="default" runId="run_fixture_tinyfable_001" />);
    expect(screen.getByRole("navigation", { name: "Distillery stages" })).toBeInTheDocument();
    expect(screen.getByTestId("stage-link-curate")).toHaveAttribute(
      "href",
      "/curate?mode=default&run=run_fixture_tinyfable_001",
    );
    expect(screen.getByTestId("stage-link-synthesize")).toHaveAttribute(
      "href",
      "/synthesize?mode=default&run=run_fixture_tinyfable_001",
    );
    expect(screen.getByTestId("stage-link-train")).toHaveAttribute(
      "href",
      "/train?mode=default&run=run_fixture_tinyfable_001",
    );
    expect(screen.getByTestId("stage-link-prove")).toHaveAttribute(
      "href",
      "/prove?mode=default&run=run_fixture_tinyfable_001",
    );
  });
});
