import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ProjectOverview } from "@/components/ProjectOverview";
import { buildStageBundle } from "@/lib/fixtures/bundle";
import { resolveAutoDistillationPlan } from "@/lib/modelPortfolio";

vi.mock("next/navigation", () => ({
  usePathname: () => "/",
  useRouter: () => ({
    replace: vi.fn(),
    refresh: vi.fn(),
  }),
  useSearchParams: () => new URLSearchParams(""),
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

describe("simple project path", () => {
  it("keeps technical terms out of the default path", () => {
    const { container } = render(
      <ProjectOverview bundle={buildStageBundle("proved")} onRefresh={vi.fn()} />,
    );
    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "What do you want your smaller model to do?",
      }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("distill-action")).toHaveTextContent(
      "Distill my model",
    );
    expect(container.textContent).not.toMatch(
      /\b(fixture|IID|OOD|recipe|SageMaker|protocol|artifact|registry|provenance)\b/i,
    );
    expect(screen.getAllByText("TinyFable Generalist").length).toBeGreaterThan(0);
    expect(screen.getByText("Current base model")).toBeInTheDocument();
  });

  it("uses bounded auto defaults for every priority", () => {
    const bundle = buildStageBundle("proved");
    for (const priority of ["quality", "speed", "cost"] as const) {
      const plan = resolveAutoDistillationPlan(bundle, priority);
      expect(plan.requestedRecipe).toBe("auto");
      expect(plan.modelPortfolioId).toBe("tinyfable_generalist");
      expect(plan.budgetCeilingUsd).toBeLessThanOrEqual(
        bundle.plan.cost.max_run_usd,
      );
    }
  });

  it("opens Advanced with an explanation for every technical setting", async () => {
    const user = userEvent.setup();
    render(
      <ProjectOverview bundle={buildStageBundle("proved")} onRefresh={vi.fn()} />,
    );
    const toggle = screen.getByTestId("advanced-toggle");
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    const settings = screen.getAllByTestId("advanced-setting");
    expect(settings.length).toBeGreaterThanOrEqual(10);
    for (const setting of settings) {
      expect(within(setting).getAllByText(/./).length).toBeGreaterThanOrEqual(3);
    }
  });

  it("switches to a specialist only after an explicit choice", async () => {
    const user = userEvent.setup();
    render(
      <ProjectOverview bundle={buildStageBundle("proved")} onRefresh={vi.fn()} />,
    );
    await user.click(screen.getByTestId("advanced-toggle"));
    const generalist = screen.getByTestId(
      "portfolio-tinyfable_generalist",
    ) as HTMLInputElement;
    const specialist = screen.getByTestId(
      "portfolio-tinyfable_transaction_specialist",
    ) as HTMLInputElement;
    expect(generalist).toBeChecked();
    await user.click(specialist);
    expect(specialist).toBeChecked();
    expect(generalist).not.toBeChecked();
  });

  it(
    "runs the saved finance example and returns a decision plus Prove action",
    async () => {
      const user = userEvent.setup();
      render(
        <ProjectOverview bundle={buildStageBundle("proved")} onRefresh={vi.fn()} />,
      );
      await user.click(screen.getByTestId("distill-action"));
      await waitFor(
        () => {
          expect(screen.getByTestId("project-result")).toBeInTheDocument();
        },
        { timeout: 6000 },
      );
      const result = screen.getByTestId("project-result");
      expect(result).toHaveTextContent("Decision");
      expect(result).toHaveTextContent("Quality");
      expect(result).toHaveTextContent("Speed");
      expect(result).toHaveTextContent("Cost");
      expect(screen.getByTestId("review-proof-action")).toHaveAttribute(
        "href",
        "/prove?mode=proved&run=run_fixture_proved_001",
      );
    },
    8000,
  );

  it("refreshes the saved project without starting work", async () => {
    const user = userEvent.setup();
    const refresh = vi.fn(async () => undefined);
    render(
      <ProjectOverview bundle={buildStageBundle("proved")} onRefresh={refresh} />,
    );
    await user.click(screen.getByTestId("distill-action"));
    await waitFor(() => screen.getByTestId("refresh-project"), { timeout: 6000 });
    await user.click(screen.getByTestId("refresh-project"));
    expect(refresh).toHaveBeenCalledTimes(1);
  }, 8000);
});
