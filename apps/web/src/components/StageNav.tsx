"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { buildStageHref, STAGES } from "@/lib/navigation";
import type { UiMode } from "@/lib/types";

export { STAGE_ROUTES } from "@/lib/navigation";

export function StageNav({ mode, runId }: { mode: UiMode; runId?: string }) {
  const pathname = usePathname();

  return (
    <nav className="stage-nav" aria-label="Distillery stages">
      {STAGES.map((stage) => {
        const current = pathname === stage.href;
        return (
          <Link
            key={stage.href}
            href={buildStageHref(stage.href, mode, runId)}
            className="stage-link"
            aria-current={current ? "page" : undefined}
            data-testid={`stage-link-${stage.name.toLowerCase()}`}
          >
            <span className="stage-index">{stage.index}</span>
            <span className="stage-name">{stage.name}</span>
          </Link>
        );
      })}
    </nav>
  );
}
