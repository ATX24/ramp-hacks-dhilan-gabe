import { defineConfig } from "@playwright/test";

const portlessUrl =
  process.env.PLAYWRIGHT_BASE_URL ?? "https://distillery-demo.localhost:1355";

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: portlessUrl,
    ignoreHTTPSErrors: true,
    trace: "on-first-retry",
  },
  webServer:
    process.env.PLAYWRIGHT_SKIP_WEBSERVER === "1"
      ? undefined
      : {
          command: "portless distillery-demo pnpm exec next dev",
          url: portlessUrl,
          reuseExistingServer: true,
          ignoreHTTPSErrors: true,
          timeout: 120_000,
        },
});
