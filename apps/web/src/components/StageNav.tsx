"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { buildProjectHref, buildStageHref, STAGES } from "@/lib/navigation";
import type { UiMode } from "@/lib/types";

export { STAGE_ROUTES } from "@/lib/navigation";

export function StageNav({ mode, runId }: { mode: UiMode; runId?: string }) {
  const pathname = usePathname();

  return (
    <nav
      className="grid grid-cols-2 gap-px overflow-hidden rounded-[14px] border border-border bg-border sm:grid-cols-3 lg:grid-cols-6"
      aria-label="Project and evidence pages"
    >
      <Link
        href={buildProjectHref(mode, runId)}
        className="grid gap-1 bg-card px-3 py-3 transition-colors hover:bg-secondary"
        aria-current={pathname === "/" ? "page" : undefined}
        data-testid="stage-link-project"
      >
        <span className="text-kicker">Start</span>
        <span className="font-serif text-base">Project</span>
      </Link>
      {STAGES.map((stage) => {
        const current = pathname === stage.href;
        return (
          <Link
            key={stage.href}
            href={buildStageHref(stage.href, mode, runId)}
            className="grid gap-1 bg-card px-3 py-3 transition-colors hover:bg-secondary aria-[current=page]:bg-[color-mix(in_oklab,var(--orange)_10%,var(--card))]"
            aria-current={current ? "page" : undefined}
            data-testid={`stage-link-${stage.id}`}
          >
            <span className="text-kicker">{stage.index}</span>
            <span className="font-serif text-base">{stage.name}</span>
          </Link>
        );
      })}
    </nav>
  );
}
