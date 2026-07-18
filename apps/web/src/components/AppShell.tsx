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
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-primary focus:px-3 focus:py-2 focus:text-primary-foreground"
      >
        Skip to content
      </a>
      <header className="site-header">
        <div className="brand-row">
          <div className="brand-block">
            <div className="brand">
              Distillery <span>· TinyFable</span>
            </div>
            <p className="tagline">Smaller models. Proven economics.</p>
          </div>
          <div
            className="rounded-full border border-border bg-card px-3 py-2 text-sm"
            aria-label="Product model"
          >
            Finance generalist <strong className="font-serif">TinyFable</strong>
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
        <ModeBanner mode={mode} />
        <details className="rounded-xl border border-border bg-card/70 px-3 py-2">
          <summary className="cursor-pointer text-sm font-medium">
            Session details · fixture mode & run reference
          </summary>
          <div className="mt-3 grid gap-3">
            <RunReferenceBar
              runId={runId}
              datasetId={datasetId}
              status={runReferenceStatus}
            />
            <Suspense fallback={null}>
              <ModeSwitcher current={mode} />
            </Suspense>
          </div>
        </details>
      </header>
      <main id="main">{children}</main>
    </div>
  );
}
