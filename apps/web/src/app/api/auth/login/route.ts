import { createHash, timingSafeEqual } from "node:crypto";
import { NextResponse, type NextRequest } from "next/server";
import { getAuthSecret, getDemoPassword } from "@/lib/auth/config";
import { safeNextPath } from "@/lib/auth/redirect";
import {
  createSessionToken,
  SESSION_COOKIE_NAME,
} from "@/lib/auth/session";

export const runtime = "nodejs";

const ATTEMPT_WINDOW_MS = 10 * 60 * 1000;
const MAX_FAILURES = 5;
const MAX_TRACKED_CLIENTS = 500;

type AttemptState = {
  failures: number;
  resetAt: number;
};

const authGlobal = globalThis as typeof globalThis & {
  distilleryAuthAttempts?: Map<string, AttemptState>;
};
const attempts =
  authGlobal.distilleryAuthAttempts ??
  (authGlobal.distilleryAuthAttempts = new Map<string, AttemptState>());

function clientKey(request: NextRequest): string {
  const forwarded = request.headers.get("x-forwarded-for")?.split(",")[0]?.trim();
  return (forwarded || request.headers.get("x-real-ip") || "unknown").slice(0, 128);
}

function pruneAttempts(now: number): void {
  for (const [key, state] of attempts) {
    if (state.resetAt <= now) attempts.delete(key);
  }
  while (attempts.size >= MAX_TRACKED_CLIENTS) {
    const oldest = attempts.keys().next().value as string | undefined;
    if (!oldest) break;
    attempts.delete(oldest);
  }
}

function attemptAllowed(key: string, now: number): boolean {
  pruneAttempts(now);
  const state = attempts.get(key);
  return !state || state.failures < MAX_FAILURES;
}

function recordFailure(key: string, now: number): void {
  const state = attempts.get(key);
  attempts.set(key, {
    failures: (state?.failures ?? 0) + 1,
    resetAt: state?.resetAt ?? now + ATTEMPT_WINDOW_MS,
  });
}

function passwordsMatch(candidate: string, expected: string): boolean {
  const candidateDigest = createHash("sha256").update(candidate).digest();
  const expectedDigest = createHash("sha256").update(expected).digest();
  return timingSafeEqual(candidateDigest, expectedDigest);
}

function failedResponse(request: NextRequest, nextPath: string): NextResponse {
  const loginUrl = new URL("/login", request.url);
  loginUrl.searchParams.set("error", "1");
  if (nextPath !== "/") loginUrl.searchParams.set("next", nextPath);
  return NextResponse.redirect(loginUrl, 303);
}

export async function POST(request: NextRequest) {
  const formData = await request.formData();
  const nextPath = safeNextPath(formData.get("next"));
  const passwordValue = formData.get("password");
  const password = typeof passwordValue === "string" ? passwordValue : "";
  const key = clientKey(request);
  const now = Date.now();

  if (!attemptAllowed(key, now)) {
    return failedResponse(request, nextPath);
  }

  let expectedPassword: string;
  let signingSecret: string;
  try {
    expectedPassword = getDemoPassword();
    signingSecret = getAuthSecret();
  } catch {
    recordFailure(key, now);
    return failedResponse(request, nextPath);
  }

  if (password.length > 1024 || !passwordsMatch(password, expectedPassword)) {
    recordFailure(key, now);
    return failedResponse(request, nextPath);
  }

  attempts.delete(key);
  const session = await createSessionToken(signingSecret, now);
  const response = NextResponse.redirect(new URL(nextPath, request.url), 303);
  response.cookies.set(SESSION_COOKIE_NAME, session.token, {
    httpOnly: true,
    secure: true,
    sameSite: "strict",
    path: "/",
    expires: session.expiresAt,
  });
  response.headers.set("Cache-Control", "no-store");
  return response;
}
