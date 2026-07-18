import { isResourceId } from "@/lib/ids";
import { UI_MODES } from "@/lib/modes";
import type { UiMode } from "@/lib/types";

const RUN_REF_KEY_PREFIX = "distillery.run_ref.";
const LEGACY_RUN_REF_KEY = "distillery.run_ref";

export type StoredRunRef = {
  mode: UiMode;
  runId: string;
  datasetId: string;
  updatedAt: string;
};

function getStorage(): Storage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function runRefKey(mode: UiMode): string {
  return `${RUN_REF_KEY_PREFIX}${mode}`;
}

function isStoredRunRef(value: unknown, mode: UiMode): value is StoredRunRef {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<StoredRunRef>;
  return (
    candidate.mode === mode &&
    isResourceId("run", candidate.runId) &&
    isResourceId("dataset", candidate.datasetId) &&
    typeof candidate.updatedAt === "string" &&
    Number.isFinite(Date.parse(candidate.updatedAt))
  );
}

export function loadRunReference(mode: UiMode): StoredRunRef | null {
  const storage = getStorage();
  if (!storage) return null;
  try {
    const raw = storage.getItem(runRefKey(mode));
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    return isStoredRunRef(parsed, mode) ? parsed : null;
  } catch {
    return null;
  }
}

export function persistRunReference(mode: UiMode, ref: StoredRunRef): boolean {
  const storage = getStorage();
  if (!storage || !isStoredRunRef(ref, mode)) return false;
  try {
    storage.setItem(runRefKey(mode), JSON.stringify(ref));
    return true;
  } catch {
    return false;
  }
}

export function clearRunReference(mode?: UiMode): void {
  const storage = getStorage();
  if (!storage) return;
  try {
    if (mode) {
      storage.removeItem(runRefKey(mode));
      return;
    }
    for (const knownMode of UI_MODES) {
      storage.removeItem(runRefKey(knownMode));
    }
    storage.removeItem(LEGACY_RUN_REF_KEY);
  } catch {
    // Storage can be unavailable in private or restricted browsing contexts.
  }
}
