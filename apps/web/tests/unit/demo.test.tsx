import { cleanup, render, screen, waitFor } from "@testing-library/react";
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
  it("keeps all five evidence routes", () => {
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

  it("renders the plain-language Demo heading", () => {
    render(
      <StageRouteContent stage="demo" bundle={buildStageBundle("proved")} />,
    );
    expect(
      screen.getByRole("heading", { level: 1, name: "Try the result" }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("demo-walkthrough")).toBeInTheDocument();
  });
});

describe("model registry and portfolio", () => {
  it("lists the base, trained candidates, and explicit specialists", () => {
    const registry = buildDemoModelRegistry(buildStageBundle("proved"));
    expect(registry.models.map((model) => model.arm_id)).toEqual([
      "student_base",
      "oracle_sft",
      "sequence_kd",
      "logit_kd",
      "ce_ablation",
      "promoted_winner",
    ]);
    expect(registry.portfolio[0]).toMatchObject({
      portfolio_id: "tinyfable_generalist",
      role: "generalist",
      recommended: true,
      selection_policy: "auto_default",
    });
    expect(
      registry.portfolio
        .filter((model) => model.role === "specialist")
        .every((model) => model.selection_policy === "explicit_only"),
    ).toBe(true);
  });

  it("does not invent trained models for a plan-only run", () => {
    const registry = buildDemoModelRegistry(buildStageBundle("default"));
    expect(registry.models.map((model) => model.arm_id)).toEqual(["student_base"]);
  });
});

describe("demo interactions", () => {
  it("exposes searchable finance examples and two honest candidates", async () => {
    const user = userEvent.setup();
    render(<DemoStage bundle={buildStageBundle("proved")} />);

    expect(screen.getByText("Current base model")).toBeInTheDocument();
    expect(screen.getByText("TinyFable Generalist")).toBeInTheDocument();
    await user.type(screen.getByTestId("demo-example-search"), "SaaS");
    expect(screen.getByText(/Review a SaaS renewal/)).toBeInTheDocument();
  });

  it("compares saved outputs and leads with a human decision", async () => {
    const user = userEvent.setup();
    render(<DemoStage bundle={buildStageBundle("proved")} />);

    await user.click(screen.getByTestId("demo-run"));
    await waitFor(() => {
      expect(screen.getByTestId("demo-decision")).toBeInTheDocument();
      expect(screen.getByTestId("demo-result-model_student_base")).toBeInTheDocument();
      expect(screen.getByTestId("demo-result-model_sequence_kd")).toBeInTheDocument();
    });
    expect(screen.getAllByText("Saved demo. Not live.").length).toBeGreaterThan(0);
    expect(screen.getByTestId("demo-decision")).toHaveTextContent("Confidence is low");
    expect(screen.getAllByText("Cost", { selector: "span" }).length).toBeGreaterThan(
      0,
    );
  });

  it("disables live output with a plain reason when no endpoint exists", () => {
    render(<DemoStage bundle={buildStageBundle("proved")} />);
    expect(screen.getByTestId("demo-infer-live")).toBeDisabled();
    expect(
      screen.getByText(/Live output stays off until an endpoint/i),
    ).toBeInTheDocument();
  });

  it("keeps technical model details in the drawer", async () => {
    const user = userEvent.setup();
    render(<DemoStage bundle={buildStageBundle("proved")} />);
    await user.click(screen.getByTestId("demo-metrics-drawer-trigger"));
    expect(screen.getByTestId("demo-metrics-drawer")).toBeInTheDocument();
    expect(screen.getByText("Training method (recipe)")).toBeInTheDocument();
    for (const row of screen.getAllByTestId("advanced-setting")) {
      expect(row).toHaveTextContent(/This |Auto |The /);
    }
  });

  it("resets and cycles the editable example", async () => {
    const user = userEvent.setup();
    render(<DemoStage bundle={buildStageBundle("proved")} />);
    const input = screen.getByTestId("demo-input") as HTMLTextAreaElement;
    const original = input.value;
    await user.clear(input);
    await user.click(input);
    await user.paste('{"patched":true}');
    await user.click(screen.getByTestId("demo-reset-example"));
    expect(input).toHaveValue(original);
    await user.click(screen.getByTestId("demo-random-example"));
    expect(input).not.toHaveValue(original);
  });
});

describe("demo gateway contract", () => {
  it("surfaces live transport errors", async () => {
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
  });

  it("refuses live output for saved-demo-only model files", async () => {
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
  it("round-trips task, models, example, and source", () => {
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
    expect(parseDemoUrlState(params, ["model_student_base"])).toEqual({
      task: "cash_reconciliation",
      modelIds: ["model_student_base", "model_sequence_kd"],
      exampleId: "ex_cash_hard_exc_001",
      runMode: "compare",
      inferenceMode: "fixture_preview",
    });
  });
});
