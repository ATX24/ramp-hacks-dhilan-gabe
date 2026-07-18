/** Render missing evidence as unknown — never invent a placeholder number. */
export function formatUnknown(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === "") return "unknown";
  return String(value);
}

export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined) return "unknown";
  return value.toLocaleString("en-US");
}

export function formatRatio(value: number | null | undefined): string {
  if (value === null || value === undefined) return "unknown";
  return `${value.toFixed(2)}×`;
}

export function formatIndex(value: number | null | undefined): string {
  if (value === null || value === undefined) return "unknown";
  return value.toFixed(3);
}

export function formatCi(
  low: number | null | undefined,
  high: number | null | undefined,
): string {
  if (low === null || low === undefined || high === null || high === undefined) {
    return "unknown";
  }
  return `[${low.toFixed(3)}, ${high.toFixed(3)}]`;
}

export function formatDurationSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined) return "unknown";
  if (value < 60) return `${value}s`;
  const minutes = Math.floor(value / 60);
  const seconds = value % 60;
  return `${minutes}m ${seconds}s`;
}

export function formatUsd(value: number | null | undefined): string {
  if (value === null || value === undefined) return "unknown";
  return `$${value.toFixed(2)}`;
}
