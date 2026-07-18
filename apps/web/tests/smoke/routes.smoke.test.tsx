import { cleanup, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StageNav } from "@/components/StageNav";
import { StageRouteContent } from "@/components/StageRouteContent";
import { buildStageBundle } from "@/lib/fixtures/bundle";
import { UI_MODES } from "@/lib/modes";
import {
  buildRootRedirect,
  resolveModeFromSearch,
  STAGES,
  STAGE_ROUTES,
} from "@/lib/navigation";

vi.mock("next/navigation", () => ({
  usePathname: () => "/curate",
  useRouter: () => ({
    replace: vi.fn(),
    refresh: vi.fn(),
  }),
  useSearchParams: () => new URLSearchParams("mode=default"),
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

describe("five-stage navigation", () => {
  it("exposes exactly five stage routes", () => {
    expect(STAGE_ROUTES).toEqual([
      "/curate",
      "/synthesize",
      "/train",
      "/prove",
      "/demo",
    ]);
  });

  it.each(UI_MODES)("preserves validated %s mode across all stage links", (mode) => {
    render(<StageNav mode={mode} />);
    for (const stage of STAGES) {
      expect(screen.getByTestId(`stage-link-${stage.id}`)).toHaveAttribute(
        "href",
        `${stage.href}?mode=${mode}`,
      );
    }
  });

  it("preserves only a supported mode in the root redirect", () => {
    expect(buildRootRedirect({ mode: "precomputed", unsafe: "discard-me" })).toBe(
      "/demo?mode=precomputed",
    );
    expect(buildRootRedirect({ mode: "default" })).toBe("/demo?mode=default");
    expect(buildRootRedirect({ unsafe: "discard-me" })).toBe("/demo");
    expect(
      buildRootRedirect({
        mode: "default",
        run: "run_fixture_tinyfable_restored_002",
        unsafe: "discard-me",
      }),
    ).toBe(
      "/demo?mode=default&run=run_fixture_tinyfable_restored_002",
    );
  });

  it("drops malformed and repeated mode parameters", () => {
    expect(buildRootRedirect({ mode: "not-a-mode" })).toBe("/demo");
    expect(buildRootRedirect({ mode: ["precomputed", "error"] })).toBe("/demo");
    expect(resolveModeFromSearch({ mode: "not-a-mode" })).toBe("default");
    expect(resolveModeFromSearch({ mode: ["precomputed", "error"] })).toBe("default");
  });
});

describe("mode × stage rendered matrix", () => {
  it.each(
    UI_MODES.flatMap((mode) =>
      STAGES.map((stage) => ({
        mode,
        stage: stage.id,
        name: stage.name,
      })),
    ),
  )("renders $mode on $stage without a blank screen", ({ mode, stage, name }) => {
    const bundle = buildStageBundle(mode);
    render(<StageRouteContent stage={stage} bundle={bundle} />);
    expect(screen.getByRole("heading", { name })).toBeInTheDocument();

    if (mode === "loading") {
      expect(screen.getByTestId("stage-loading")).toBeInTheDocument();
    } else if (mode === "fetch_failure") {
      expect(screen.getByTestId("stage-fetch-failure")).toHaveAttribute("role", "alert");
    }
  });
});

describe("cross-stage state honesty", () => {
  it("renders insufficient evidence as a prior completion, never preparation-only", () => {
    const bundle = buildStageBundle("insufficient_evidence");
    render(<StageRouteContent stage="train" bundle={bundle} />);
    expect(screen.getByTestId("run-presentation")).toHaveAttribute(
      "data-presentation",
      "prior_completion",
    );
    expect(screen.getByText("Precomputed prior completion")).toBeInTheDocument();
    expect(screen.queryByText("No training yet")).not.toBeInTheDocument();
    expect(screen.getByTestId("job-activity")).toHaveTextContent("none");
  });

  it.each(UI_MODES)("hides cancellation for %s fixture mode", (mode) => {
    const bundle = buildStageBundle(mode);
    render(<StageRouteContent stage="train" bundle={bundle} />);
    if (mode === "loading" || mode === "fetch_failure") {
      expect(screen.queryByTestId("cancel-button")).not.toBeInTheDocument();
      return;
    }
    expect(screen.queryByTestId("cancel-button")).not.toBeInTheDocument();
    expect(screen.getByTestId("cancellation-unavailable")).toBeInTheDocument();
  });

  it("labels every precomputed system row as prior-run measurement", () => {
    const bundle = buildStageBundle("precomputed");
    render(<StageRouteContent stage="prove" bundle={bundle} />);
    expect(
      screen.getAllByText("Prior-run precomputed measurement"),
    ).toHaveLength(7);
  });

  it.each([
    ["proved", "proved"],
    ["do_not_distill", "do_not_distill"],
    ["failed_quality", "failed_quality"],
    ["failed_economics", "failed_economics"],
    ["insufficient_evidence", "insufficient_evidence"],
  ] as const)("renders %s proof status from prior-run provenance", (mode, status) => {
    const bundle = buildStageBundle(mode);
    expect(bundle.proof?.proof_status).toBe(status);
    expect(bundle.proof?.precomputed).toBe(true);
    expect(bundle.artifact?.precomputed).toBe(true);
    render(<StageRouteContent stage="prove" bundle={bundle} />);
    expect(screen.getByText(status)).toBeInTheDocument();
  });
});
