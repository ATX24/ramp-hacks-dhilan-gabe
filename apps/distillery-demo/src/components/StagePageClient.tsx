"use client";

import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import type { RunReferenceStatus } from "@/components/RunReferenceBar";
import { StageRouteContent } from "@/components/StageRouteContent";
import { createApiClient } from "@/lib/api";
import { getDefaultRunId } from "@/lib/fixtures/catalog";
import { withStageLoadFailure } from "@/lib/loadStage";
import type { RunSelection } from "@/lib/navigation";
import {
  loadRunReference,
  persistRunReference,
  type StoredRunRef,
} from "@/lib/storage";
import type { StageBundle, StageId } from "@/lib/types";

export function StagePageClient({
  stage,
  initialBundle,
  runSelection,
}: {
  stage: StageId;
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

    async function reconstructSelection(): Promise<void> {
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
        const selectedBundle = await client.loadStage();
        if (cancelled) return;

        setBundle(selectedBundle);
        setResolved(true);
        const reference: StoredRunRef = {
          mode: initialBundle.mode,
          runId: selectedBundle.run.run_id,
          datasetId: selectedBundle.dataset.dataset_id,
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

    void reconstructSelection();
    return () => {
      cancelled = true;
    };
  }, [initialBundle.mode, runSelection]);

  const visibleBundle: StageBundle = resolved
    ? bundle
    : {
        ...bundle,
        load_state: {
          status: "loading",
          message: "Restoring this mode’s stored run reference from local storage.",
        },
      };

  return (
    <AppShell
      mode={visibleBundle.mode}
      runId={visibleBundle.run.run_id}
      datasetId={visibleBundle.dataset.dataset_id}
      runReferenceStatus={referenceStatus}
    >
      <StageRouteContent stage={stage} bundle={visibleBundle} />
    </AppShell>
  );
}
