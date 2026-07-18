"use client";

import { usePathname, useRouter } from "next/navigation";
import { isUiMode, UI_MODES } from "@/lib/modes";
import { buildModeHref } from "@/lib/navigation";
import type { UiMode } from "@/lib/types";

const MODE_LABELS: Record<UiMode, string> = {
  default: "Basic sample",
  precomputed: "Saved run",
  proved: "Saved passing result",
  do_not_distill: "Keep current model",
  failed_quality: "Accuracy missed",
  failed_economics: "Cost target missed",
  error: "Data check error",
  unavailable: "Method unavailable",
  insufficient_evidence: "More proof needed",
  skipped_synthesis: "Answers already present",
  no_training_yet: "Plan only",
  loading: "Loading",
  fetch_failure: "Load error",
};

export function ModeSwitcher({ current }: { current: UiMode }) {
  const router = useRouter();
  const pathname = usePathname();

  return (
    <div className="grid gap-1.5">
      <div className="mode-bar">
        <label htmlFor="ui-mode">Saved sample</label>
        <select
          id="ui-mode"
          value={current}
          aria-label="Choose a saved sample state"
          aria-describedby="ui-mode-help"
          data-testid="mode-switcher"
          onChange={(event) => {
            const next = event.target.value;
            if (!isUiMode(next)) return;
            router.replace(buildModeHref(pathname, next));
            router.refresh();
          }}
        >
          {UI_MODES.map((mode) => (
            <option key={mode} value={mode}>
              {MODE_LABELS[mode]}
            </option>
          ))}
        </select>
      </div>
      <p id="ui-mode-help" className="text-sm text-muted-foreground">
        This changes the saved state shown on the evidence pages. Leave the passing
        result selected for the judge path. It never starts a live job.
      </p>
    </div>
  );
}
