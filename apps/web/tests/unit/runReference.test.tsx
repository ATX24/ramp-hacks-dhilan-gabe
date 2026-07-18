import { cleanup, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StagePageClient } from "@/components/StagePageClient";
import { buildStageBundle } from "@/lib/fixtures/bundle";
import {
  DEFAULT_DATASET_ID,
  DEFAULT_RUN_ID,
  FIXTURE_MODE_CATALOG,
  RESTORED_RUN_ID,
} from "@/lib/fixtures/catalog";
import {
  clearRunReference,
  loadRunReference,
  persistRunReference,
} from "@/lib/storage";

const navigation = vi.hoisted(() => ({
  replace: vi.fn(),
  refresh: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  usePathname: () => "/train",
  useRouter: () => navigation,
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
  clearRunReference();
  navigation.replace.mockReset();
  navigation.refresh.mockReset();
});

describe("rendered run-reference reconstruction", () => {
  it("restores a valid stored run instead of clobbering it with the default", async () => {
    persistRunReference("default", {
      mode: "default",
      runId: RESTORED_RUN_ID,
      datasetId: DEFAULT_DATASET_ID,
      updatedAt: "2026-07-18T12:00:00.000Z",
    });

    render(
      <StagePageClient
        stage="train"
        initialBundle={buildStageBundle("default")}
        runSelection={{ kind: "absent" }}
      />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("run-reference")).toHaveTextContent(RESTORED_RUN_ID);
    });
    expect(screen.getByTestId("stage-link-prove")).toHaveAttribute(
      "href",
      `/prove?mode=default&run=${RESTORED_RUN_ID}`,
    );
  });

  it("lets an explicit run win and survive rendered refresh reconstruction", async () => {
    persistRunReference("default", {
      mode: "default",
      runId: RESTORED_RUN_ID,
      datasetId: DEFAULT_DATASET_ID,
      updatedAt: "2026-07-18T12:00:00.000Z",
    });

    const first = render(
      <StagePageClient
        stage="train"
        initialBundle={buildStageBundle("default", DEFAULT_RUN_ID)}
        runSelection={{ kind: "valid", runId: DEFAULT_RUN_ID }}
      />,
    );
    await waitFor(() => {
      expect(loadRunReference("default")?.runId).toBe(DEFAULT_RUN_ID);
    });
    first.unmount();

    render(
      <StagePageClient
        stage="prove"
        initialBundle={buildStageBundle("default")}
        runSelection={{ kind: "absent" }}
      />,
    );
    await waitFor(() => {
      expect(screen.getByTestId("run-reference")).toHaveTextContent(DEFAULT_RUN_ID);
    });
  });

  it("keeps run references isolated by mode", async () => {
    persistRunReference("default", {
      mode: "default",
      runId: RESTORED_RUN_ID,
      datasetId: DEFAULT_DATASET_ID,
      updatedAt: "2026-07-18T12:00:00.000Z",
    });

    render(
      <StagePageClient
        stage="prove"
        initialBundle={buildStageBundle("proved")}
        runSelection={{ kind: "absent" }}
      />,
    );
    await waitFor(() => {
      expect(screen.getByTestId("run-reference")).toHaveTextContent(
        FIXTURE_MODE_CATALOG.proved.defaultRunId,
      );
    });
    expect(loadRunReference("default")?.runId).toBe(RESTORED_RUN_ID);
  });

  it("surfaces an unknown stored run without substituting the default", async () => {
    const unknownRunId = "run_fixture_unknown_999";
    persistRunReference("default", {
      mode: "default",
      runId: unknownRunId,
      datasetId: DEFAULT_DATASET_ID,
      updatedAt: "2026-07-18T12:00:00.000Z",
    });

    render(
      <StagePageClient
        stage="train"
        initialBundle={buildStageBundle("default")}
        runSelection={{ kind: "absent" }}
      />,
    );
    await waitFor(() => {
      expect(screen.getByTestId("stage-fetch-failure")).toHaveTextContent(
        "RESOURCE_NOT_FOUND",
      );
    });
    expect(screen.getByTestId("run-reference")).toHaveTextContent(unknownRunId);
    expect(loadRunReference("default")?.runId).toBe(unknownRunId);
  });
});
