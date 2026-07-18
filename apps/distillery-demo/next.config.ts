import type { NextConfig } from "next";
import { PHASE_PRODUCTION_BUILD } from "next/constants";

const REQUIRED_BUILD_SECRETS = [
  "DISTILLERY_DEMO_PASSWORD",
  "DISTILLERY_AUTH_SECRET",
] as const;

export default function nextConfig(phase: string): NextConfig {
  if (phase === PHASE_PRODUCTION_BUILD) {
    const missing = REQUIRED_BUILD_SECRETS.filter((name) => !process.env[name]);
    if (missing.length > 0) {
      throw new Error(
        `Distillery authentication is not configured. Set ${missing.join(" and ")} before running a production build.`,
      );
    }
  }

  return {
    reactStrictMode: true,
    poweredByHeader: false,
  };
}
