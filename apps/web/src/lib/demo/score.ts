function stableStringify(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(",")}]`;
  }
  const record = value as Record<string, unknown>;
  const keys = Object.keys(record).sort();
  return `{${keys.map((key) => `${JSON.stringify(key)}:${stableStringify(record[key])}`).join(",")}}`;
}

/**
 * Exact-match joint score when gold is available.
 * Returns null when gold is missing — UI must not invent a score.
 */
export function scoreAgainstGold(
  structuredOutput: Record<string, unknown>,
  goldOutput: Record<string, unknown> | null,
): { score: number | null; detail: string | null } {
  if (goldOutput === null) {
    return { score: null, detail: "No gold available for this example." };
  }
  const match = stableStringify(structuredOutput) === stableStringify(goldOutput);
  return {
    score: match ? 1 : 0,
    detail: match ? "Exact match to gold." : "Does not exactly match gold.",
  };
}
