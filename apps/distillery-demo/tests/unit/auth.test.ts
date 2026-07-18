import { describe, expect, it } from "vitest";
import {
  createSessionToken,
  SESSION_TTL_SECONDS,
  verifySessionToken,
} from "@/lib/auth/session";

const TEST_SECRET = "test-only-signing-secret-with-at-least-32-characters";
const OTHER_TEST_SECRET = "another-test-signing-secret-with-at-least-32-chars";
const NOW = Date.UTC(2026, 6, 18, 12, 0, 0);

describe("signed demo sessions", () => {
  it("accepts a valid unexpired token", async () => {
    const { token } = await createSessionToken(TEST_SECRET, NOW);

    await expect(verifySessionToken(token, TEST_SECRET, NOW)).resolves.toBe(true);
  });

  it("rejects tampered and incorrectly signed tokens", async () => {
    const { token } = await createSessionToken(TEST_SECRET, NOW);
    const parts = token.split(".");
    const signature = parts[3] ?? "";
    parts[3] = `${signature.startsWith("A") ? "B" : "A"}${signature.slice(1)}`;

    await expect(
      verifySessionToken(parts.join("."), TEST_SECRET, NOW),
    ).resolves.toBe(false);
    await expect(
      verifySessionToken(token, OTHER_TEST_SECRET, NOW),
    ).resolves.toBe(false);
  });

  it("rejects an expired token", async () => {
    const { token } = await createSessionToken(TEST_SECRET, NOW);
    const afterExpiry = NOW + (SESSION_TTL_SECONDS + 1) * 1000;

    await expect(
      verifySessionToken(token, TEST_SECRET, afterExpiry),
    ).resolves.toBe(false);
  });
});
