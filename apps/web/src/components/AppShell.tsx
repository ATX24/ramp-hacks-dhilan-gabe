"use client";

import Link from "next/link";
import { LogOut } from "lucide-react";
import type { ReactNode } from "react";
import { ModeBanner } from "@/components/ModeBanner";
import {
  RunReferenceBar,
  type RunReferenceStatus,
} from "@/components/RunReferenceBar";
import { StageNav } from "@/components/StageNav";
import { Button } from "@/components/ui/button";
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
          <Link href="/" className="brand-block" aria-label="Distillery home">
            <p className="text-kicker text-muted-foreground">
              Distillation workspace
            </p>
            <div className="brand">
              Distillery
            </div>
            <p className="tagline">
              Teach a smaller model. Test it on finance work.
            </p>
          </Link>
          <div className="flex items-center gap-2">
            <div
              className="hidden rounded-full border border-black/15 bg-card px-3 py-2 text-sm sm:block"
              aria-label="Product model"
            >
              Finance model <strong className="font-serif">TinyFable</strong>
            </div>
            <form action="/api/auth/logout" method="post">
              <Button type="submit" variant="outline" size="lg">
                <LogOut aria-hidden />
                Sign out
              </Button>
            </form>
          </div>
        </div>
      </header>
      <main id="main" className="central-main">
        {children}
      </main>
      <aside className="mt-8 border-t border-black/15 pt-4">
        <details className="rounded-[20px] border border-black/15 bg-card px-4 py-3">
          <summary className="cursor-pointer text-sm font-medium">
            More stages and session details
          </summary>
          <div className="mt-4 grid gap-3">
            <StageNav
              mode={mode}
              runId={
                runReferenceStatus === "resolving" ||
                runReferenceStatus === "invalid"
                  ? undefined
                  : runId
              }
            />
            <ModeBanner mode={mode} />
            <RunReferenceBar
              runId={runId}
              datasetId={datasetId}
              status={runReferenceStatus}
            />
          </div>
        </details>
      </aside>
    </div>
  );
}
