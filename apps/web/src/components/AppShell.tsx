"use client";

import { Suspense, type ReactNode } from "react";
import { ModeBanner } from "@/components/ModeBanner";
import { ModeSwitcher } from "@/components/ModeSwitcher";
import {
  RunReferenceBar,
  type RunReferenceStatus,
} from "@/components/RunReferenceBar";
import { StageNav } from "@/components/StageNav";
import type { UiMode } from "@/lib/types";

export function AppShell({
  mode,
  runId,
  datasetId,
  runReferenceStatus,
  children,
}: {
  mode: UiMode;
  runId: string;
  datasetId: string;
  runReferenceStatus: RunReferenceStatus;
  children: ReactNode;
}) {
  return (
    <div className="shell">
      <header className="site-header">
        <div className="brand-row">
          <div className="brand-block">
            <div className="brand">
              Distillery <span>· TinyFable</span>
            </div>
            <p className="tagline">Smaller models. Proven economics.</p>
          </div>
          <div className="model-chip" aria-label="Product model">
            Finance generalist <strong>TinyFable</strong>
          </div>
        </div>
        <StageNav
          mode={mode}
          runId={
            runReferenceStatus === "resolving" || runReferenceStatus === "invalid"
              ? undefined
              : runId
          }
        />
        <RunReferenceBar
          runId={runId}
          datasetId={datasetId}
          status={runReferenceStatus}
        />
        <ModeBanner mode={mode} />
        <Suspense fallback={null}>
          <ModeSwitcher current={mode} />
        </Suspense>
      </header>
      <main id="main">{children}</main>
    </div>
  );
}
