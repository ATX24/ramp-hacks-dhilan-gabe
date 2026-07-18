"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { ModeBanner } from "@/components/ModeBanner";
import {
  RunReferenceBar,
  type RunReferenceStatus,
} from "@/components/RunReferenceBar";
import { StageNav } from "@/components/StageNav";
import { Badge } from "@/components/ui/badge";
import type { UiMode } from "@/lib/types";

export function AppShell({
  mode,
  runId,
  datasetId,
  runReferenceStatus,
  overview = false,
  children,
}: {
  mode: UiMode;
  runId: string;
  datasetId: string;
  runReferenceStatus: RunReferenceStatus;
  overview?: boolean;
  children: ReactNode;
}) {
  return (
    <div className="shell">
      <a
        href="#main"
        className="sr-only z-50 rounded-full bg-primary px-4 py-2 text-primary-foreground focus:not-sr-only focus:fixed focus:top-3 focus:left-3"
      >
        Skip to content
      </a>
      <header className="site-header">
        <div className="brand-row">
          <div className="brand-block">
            <Link className="brand" href="/" aria-label="Open the project overview">
              Distillery <span>· TinyFable</span>
            </Link>
            <p className="tagline">Smaller models. Proven economics.</p>
          </div>
          <Badge
            variant="outline"
            className="rounded-full bg-card px-3 py-1.5 font-normal"
            aria-label="Recommended model"
          >
            Recommended
            <strong className="ml-1 font-serif font-normal text-[var(--orange)]">
              TinyFable Generalist
            </strong>
          </Badge>
        </div>
        {!overview ? (
          <StageNav
            mode={mode}
            runId={
              runReferenceStatus === "resolving" || runReferenceStatus === "invalid"
                ? undefined
                : runId
            }
          />
        ) : null}
        {!overview ? (
          <>
            <RunReferenceBar
              runId={runId}
              datasetId={datasetId}
              status={runReferenceStatus}
            />
            <ModeBanner mode={mode} />
          </>
        ) : null}
      </header>
      <main id="main" tabIndex={-1}>
        {children}
      </main>
    </div>
  );
}
