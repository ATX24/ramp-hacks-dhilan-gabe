"use client";

import { usePathname, useRouter } from "next/navigation";
import { isUiMode, UI_MODES } from "@/lib/modes";
import { buildModeHref } from "@/lib/navigation";
import type { UiMode } from "@/lib/types";

export function ModeSwitcher({ current }: { current: UiMode }) {
  const router = useRouter();
  const pathname = usePathname();

  return (
    <div className="mode-bar">
      <label htmlFor="ui-mode">Fixture mode</label>
      <select
        id="ui-mode"
        value={current}
        aria-label="Fixture mode"
        onChange={(event) => {
          const next = event.target.value;
          if (!isUiMode(next)) return;
          router.replace(buildModeHref(pathname, next));
          router.refresh();
        }}
      >
        {UI_MODES.map((mode) => (
          <option key={mode} value={mode}>
            {mode}
          </option>
        ))}
      </select>
    </div>
  );
}
