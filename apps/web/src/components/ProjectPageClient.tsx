"use client";

import { useCallback, useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { ProjectOverview } from "@/components/ProjectOverview";
import type { RunReferenceStatus } from "@/components/RunReferenceBar";
import { createApiClient } from "@/lib/api";
import { getDefaultRunId } from "@/lib/fixtures/catalog";
import { withStageLoadFailure } from "@/lib/loadStage";
import type { RunSelection } from "@/lib/navigation";
import {
  loadRunReference,
  persistRunReference,
  type StoredRunRef,
} from "@/lib/storage";
import type { StageBundle } from "@/lib/types";

export function ProjectPageClient({
  initialBundle,
  runSelection,
}: {
  initialBundle: StageBundle;
  runSelection: RunSelection;
}) {
  const [bundle, setBundle] = useState(initialBundle);
  const [resolved, setResolved] = useState(runSelection.kind !== "absent");
  const [referenceStatus, setReferenceStatus] = useState<RunReferenceStatus>(
    runSelection.kind === "absent"
      ? "resolving"
      : runSelection.kind === "invalid"
        ? "invalid"
        : "session_only",
  );

  useEffect(() => {
    let cancelled = false;

    async function restoreProject(): Promise<void> {
      if (runSelection.kind === "invalid") {
        setReferenceStatus("invalid");
        return;
      }

      const stored =
        runSelection.kind === "absent" ? loadRunReference(initialBundle.mode) : null;
      const selectedRunId =
        runSelection.kind === "valid"
          ? runSelection.runId
          : stored?.runId ?? getDefaultRunId(initialBundle.mode);

      try {
        const client = createApiClient({
          mode: initialBundle.mode,
          runId: selectedRunId,
        });
        if (stored) {
          await client.getDataset(stored.datasetId);
        }
        const next = await client.loadStage();
        if (cancelled) return;

        setBundle(next);
        setResolved(true);
        const reference: StoredRunRef = {
          mode: initialBundle.mode,
          runId: next.run.run_id,
          datasetId: next.dataset.dataset_id,
          updatedAt: new Date().toISOString(),
        };
        setReferenceStatus(
          persistRunReference(initialBundle.mode, reference)
            ? "stored"
            : "session_only",
        );
      } catch (error) {
        if (cancelled) return;
        setBundle(withStageLoadFailure(initialBundle.mode, selectedRunId, error));
        setResolved(true);
        setReferenceStatus("session_only");
      }
    }

    void restoreProject();
    return () => {
      cancelled = true;
    };
  }, [initialBundle.mode, runSelection]);

  const refresh = useCallback(async () => {
    const client = createApiClient({
      mode: bundle.mode,
      runId: bundle.run.run_id,
    });
    try {
      setBundle(await client.loadStage());
    } catch (error) {
      setBundle(withStageLoadFailure(bundle.mode, bundle.run.run_id, error));
    }
  }, [bundle.mode, bundle.run.run_id]);

  const visibleBundle: StageBundle = resolved
    ? bundle
    : {
        ...bundle,
        load_state: {
          status: "loading",
          message: "Opening the saved project in this browser.",
        },
      };

  return (
    <AppShell
      mode={visibleBundle.mode}
      runId={visibleBundle.run.run_id}
      datasetId={visibleBundle.dataset.dataset_id}
      runReferenceStatus={referenceStatus}
      overview
    >
      <ProjectOverview bundle={visibleBundle} onRefresh={refresh} />
    </AppShell>
  );
}
