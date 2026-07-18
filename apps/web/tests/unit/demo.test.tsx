import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DemoStage } from "@/components/stages/DemoStage";
import { StageNav, STAGE_ROUTES } from "@/components/StageNav";
import { StageRouteContent } from "@/components/StageRouteContent";
import { createDemoGateway } from "@/lib/demo/gateway";
import { buildDemoModelRegistry } from "@/lib/demo/registry";
import { parseDemoUrlState, serializeDemoUrlState } from "@/lib/demo/urlState";
import { buildStageBundle } from "@/lib/fixtures/bundle";

const replaceMock = vi.fn();

vi.mock("next/navigation", () => ({
  usePathname: () => "/demo",
  useRouter: () => ({
    replace: replaceMock,
    refresh: vi.fn(),
  }),
  useSearchParams: () => new URLSearchParams("mode=proved"),
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
  replaceMock.mockReset();
});

describe("demo navigation", () => {
  it("exposes Demo as the fifth primary stage", () => {
    expect(STAGE_ROUTES).toEqual([
      "/curate",
      "/synthesize",
      "/train",
      "/prove",
      "/demo",
    ]);
    render(<StageNav mode="proved" runId="run_fixture_proved_001" />);
    expect(screen.getByTestId("stage-link-demo")).toHaveAttribute(
      "href",
      "/demo?mode=proved&run=run_fixture_proved_001",
    );
  });

  it("renders Demo inside the stage route matrix", () => {
    const bundle = buildStageBundle("proved");
    render(<StageRouteContent stage="demo" bundle={bundle} />);
    expect(screen.getByRole("heading", { level: 1 })).toBeInTheDocument();
    expect(screen.getByTestId("demo-hero")).toBeInTheDocument();
    expect(screen.getByText("Not live inference")).toBeInTheDocument();
  });
});

describe("demo model registry", () => {
  it("lists base student plus trained arms from the registry payload", () => {
    const registry = buildDemoModelRegistry(buildStageBundle("proved"));
    const arms = registry.models.map((model) => model.arm_id);
    expect(arms).toEqual([
      "student_base",
      "oracle_sft",
      "sequence_kd",
      "logit_kd",
      "ce_ablation",
      "promoted_winner",
    ]);
  });

  it("keeps the two honest saved-data arms available by default", () => {
    const bundle = buildStageBundle("default");
    const registry = buildDemoModelRegistry(bundle);
    expect(registry.models.map((model) => model.arm_id)).toEqual([
      "student_base",
      "sequence_kd",
    ]);
  });

  it("keeps detailed proof statistics out of the primary demo flow", () => {
    const bundle = buildStageBundle("proved");
    render(<DemoStage bundle={bundle} />);
    expect(screen.queryByTestId("demo-stats-panel")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "saved demo data",
    );
  });
});

describe("demo playground interactions", () => {
  it("selects an example and prefills the plain-language input without running", async () => {
    const user = userEvent.setup();
    render(<DemoStage bundle={buildStageBundle("proved")} />);

    const preset = screen.getByTestId("example-preset-budget-miss");
    await user.click(preset);
    expect(preset).toHaveAttribute("aria-pressed", "true");
    expect(
      (screen.getByTestId("demo-plain-input") as HTMLTextAreaElement).value,
    ).toMatch(/budget/i);
    expect(within(screen.getByTestId("demo-results")).queryAllByRole("article")).toHaveLength(
      0,
    );
  });

  it("compares the original and taught saved-data arms", async () => {
    const user = userEvent.setup();
    render(<DemoStage bundle={buildStageBundle("proved")} />);

    await user.click(screen.getByTestId("demo-run"));

    await waitFor(() => {
      expect(
        within(screen.getByTestId("demo-results")).getAllByRole("article"),
      ).toHaveLength(2);
    });
    for (const card of within(screen.getByTestId("demo-results")).getAllByRole(
      "article",
    )) {
      expect(card).toHaveAttribute("data-provenance", "fixture_preview");
    }
  });

  it("does not expose or imply live inference controls", async () => {
    const user = userEvent.setup();
    render(<DemoStage bundle={buildStageBundle("proved")} />);

    expect(screen.queryByText("Live model inference")).not.toBeInTheDocument();
    await user.click(screen.getByTestId("demo-run"));

    await waitFor(() => {
      expect(
        within(screen.getByTestId("demo-results")).getAllByText(
          "Saved demo data. Not live inference.",
        ),
      ).toHaveLength(2);
    });
  });

  it("labels every comparison result as saved demo data", async () => {
    const user = userEvent.setup();
    render(<DemoStage bundle={buildStageBundle("proved")} />);
    await user.click(screen.getByTestId("demo-run"));
    await waitFor(() => {
      expect(
        within(screen.getByTestId("demo-results")).getAllByText(
          "Saved demo data",
        ),
      ).toHaveLength(2);
    });
  });

  it("keeps manual edits and hides raw JSON behind disclosure", async () => {
    const user = userEvent.setup();
    render(<DemoStage bundle={buildStageBundle("proved")} />);
    const input = screen.getByTestId("demo-plain-input");
    await user.clear(input);
    await user.type(input, "Keep this manual edit.");
    expect(input).toHaveValue("Keep this manual edit.");
    expect(screen.getByText("Edit raw example").closest("details")).not.toHaveAttribute(
      "open",
    );
    expect(within(screen.getByTestId("demo-results")).queryAllByRole("article")).toHaveLength(
      0,
    );
  });
});

describe("demo gateway contract", () => {
  it("calls the typed live endpoint and surfaces transport errors", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new Error("connection refused");
    });
    const gateway = createDemoGateway({
      liveBaseUrl: "http://127.0.0.1:9999",
      fetchImpl,
    });
    const registry = buildDemoModelRegistry(buildStageBundle("proved"));
    const liveModel = {
      ...registry.models.find((model) => model.arm_id === "sequence_kd")!,
      serving: {
        availability: "live" as const,
        endpoint_id: "ep_demo_001",
        artifact_id: "art_tinyfable_proved_001",
        reason: null,
      },
    };
    const liveRegistry = {
      ...registry,
      models: registry.models.map((model) =>
        model.model_id === liveModel.model_id ? liveModel : model,
      ),
    };

    const response = await gateway.infer(liveRegistry, {
      model_id: liveModel.model_id,
      task: "transaction_review",
      example_id: "ex_txn_hard_001",
      input: { amount_minor: 1 },
      mode: "live",
    });

    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:9999/v1/demo/infer",
      expect.objectContaining({ method: "POST" }),
    );
    expect(response.status).toBe("error");
    if (response.status === "error") {
      expect(response.code).toBe("LIVE_TRANSPORT_ERROR");
    }
  });

  it("refuses to invent live output for fixture_preview-only artifacts", async () => {
    const gateway = createDemoGateway({
      liveBaseUrl: "http://127.0.0.1:9999",
      fetchImpl: vi.fn(),
    });
    const registry = buildDemoModelRegistry(buildStageBundle("proved"));
    const response = await gateway.infer(registry, {
      model_id: "model_sequence_kd",
      task: "transaction_review",
      example_id: "ex_txn_hard_001",
      input: { amount_minor: 1 },
      mode: "live",
    });
    expect(response.status).toBe("unavailable");
    if (response.status === "unavailable") {
      expect(response.code).toBe("ARTIFACT_NOT_SERVABLE");
    }
  });
});

describe("demo URL state", () => {
  it("round-trips task/models/example share state", () => {
    const params = serializeDemoUrlState(
      {
        task: "cash_reconciliation",
        modelIds: ["model_student_base", "model_sequence_kd"],
        exampleId: "ex_cash_hard_exc_001",
        runMode: "compare",
        inferenceMode: "fixture_preview",
      },
      { mode: "proved", runId: "run_fixture_proved_001" },
    );
    const parsed = parseDemoUrlState(params, ["model_student_base"]);
    expect(parsed).toEqual({
      task: "cash_reconciliation",
      modelIds: ["model_student_base", "model_sequence_kd"],
      exampleId: "ex_cash_hard_exc_001",
      runMode: "compare",
      inferenceMode: "fixture_preview",
    });
    expect(params.get("mode")).toBe("proved");
    expect(params.get("run")).toBe("run_fixture_proved_001");
  });
});
