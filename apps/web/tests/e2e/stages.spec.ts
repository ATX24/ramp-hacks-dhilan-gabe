import { expect, test } from "@playwright/test";

const ROUTES = [
  ["/curate?mode=proved", "Check the data"],
  ["/synthesize?mode=proved", "Fill the missing answers"],
  ["/train?mode=proved", "Make the smaller model"],
  ["/prove?mode=proved", "Check the result"],
  ["/demo?mode=proved", "Try the result"],
] as const;

test.describe("judge path acceptance", () => {
  test("explains the product in five seconds", async ({ page }) => {
    await page.goto("/");
    await expect(
      page.getByRole("heading", {
        level: 1,
        name: "What do you want your smaller model to do?",
      }),
    ).toBeVisible();
    await expect(page.getByTestId("distill-action")).toHaveText(
      "Distill my model",
    );
    await expect(page.getByText("Current base model")).toBeVisible();
    await expect(page.getByText("TinyFable Generalist").first()).toBeVisible();
    await expect(page.getByTestId("mode-switcher")).toHaveCount(0);
    await expect(page.locator("h1")).toHaveCount(1);
  });

  test("completes the default path without internal mode controls", async ({
    page,
  }) => {
    await page.goto("/");
    await page.getByTestId("distill-action").click();
    const result = page.getByTestId("project-result");
    await expect(result).toBeVisible({ timeout: 10_000 });
    await expect(result).toContainText("Decision");
    await expect(result).toContainText("Why");
    await expect(result).toContainText("Confidence");
    await expect(result).toContainText("Quality");
    await expect(result).toContainText("Speed");
    await expect(result).toContainText("Cost");
    await expect(result.getByText("Saved demo. Not live.")).toHaveCount(2);

    await Promise.all([
      page.waitForURL(/\/prove\?mode=proved/),
      page.getByTestId("review-proof-action").click(),
    ]);
    await expect(
      page.getByRole("heading", { level: 1, name: "Check the result" }),
    ).toBeVisible();
    await expect(page.getByTestId("proof-decision")).toBeVisible();
    await expect(page.getByTestId("mode-switcher")).toHaveCount(0);
  });

  test("keeps the goal and primary action in the 390 by 844 fold", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");
    const header = await page.locator("header").boundingBox();
    const action = await page.getByTestId("distill-action").boundingBox();
    expect(header).not.toBeNull();
    expect(action).not.toBeNull();
    expect(header!.height).toBeLessThanOrEqual(844 * 0.4);
    expect(action!.y + action!.height).toBeLessThanOrEqual(844);

    const pageWidth = await page.evaluate(
      () => document.documentElement.scrollWidth,
    );
    expect(pageWidth).toBeLessThanOrEqual(390);
  });

  test("keeps jargon out until Advanced opens", async ({ page }) => {
    await page.goto("/");
    const text = await page.locator("main").innerText();
    expect(text).not.toMatch(
      /\b(fixture|IID|OOD|recipe|SageMaker|protocol|artifact|registry|provenance)\b/i,
    );
    await page.getByTestId("advanced-toggle").click();
    await expect(page.getByText("Training method (recipe)")).toBeVisible();
    const settings = page.getByTestId("advanced-setting");
    expect(await settings.count()).toBeGreaterThanOrEqual(10);
    for (let index = 0; index < (await settings.count()); index += 1) {
      const settingText = (await settings.nth(index).innerText()).trim();
      expect(settingText.split("\n").filter(Boolean).length).toBeGreaterThanOrEqual(3);
    }

    for (const [route] of ROUTES) {
      await page.goto(route);
      const routeText = await page.locator("main").innerText();
      expect(routeText).not.toMatch(
        /\b(fixture|IID|OOD|recipe|SageMaker|protocol|artifact|registry|provenance|sha256|hash)\b|\b(run|data|result)\s+ID\b/i,
      );
    }
  });

  test("disables live output with a plain reason", async ({ page }) => {
    await page.goto("/demo?mode=proved");
    await expect(page.getByTestId("demo-infer-live")).toBeDisabled();
    await expect(
      page.getByText(/Live output stays off until an endpoint/i),
    ).toBeVisible();
    await expect(page.getByText("Saved demo outputs ready")).toBeVisible();
  });

  test("leads result cards with the decision and hides raw JSON", async ({
    page,
  }) => {
    await page.goto("/demo?mode=proved");
    await page.getByTestId("demo-run").click();
    await expect(page.getByTestId("demo-decision")).toBeVisible();
    const first = page
      .locator("[data-testid^='demo-result-model_'][data-status='ok']")
      .first();
    await expect(first.getByText("Decision")).toBeVisible();
    await expect(first.getByText("Why")).toBeVisible();
    await expect(first.getByText("Confidence", { exact: true })).toBeVisible();
    await expect(first.getByText("Raw result (JSON)")).not.toBeVisible();
    await first.getByText("Advanced output details").click();
    await expect(first.getByText("Raw result (JSON)")).toBeVisible();
  });

  test("supports the skip link and reaches the primary action quickly", async ({
    page,
  }) => {
    await page.goto("/");
    const skip = page.getByRole("link", { name: "Skip to content" });
    await page.keyboard.press("Tab");
    await expect(skip).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(page.locator("main")).toBeFocused();

    await page.reload();
    let reachedAction = false;
    for (let tabs = 0; tabs < 8; tabs += 1) {
      await page.keyboard.press("Tab");
      reachedAction = await page
        .getByTestId("distill-action")
        .evaluate((element) => element === document.activeElement);
      if (reachedAction) break;
    }
    expect(reachedAction).toBe(true);
  });

  test("places a clear error next to the goal input", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("project-goal").fill("short");
    await page.getByTestId("distill-action").click();
    const alert = page.locator("main [role='alert']");
    await expect(alert).toHaveText(/Tell us a little more/);
    expect(
      await page.getByTestId("distill-action").evaluate((button) => {
        const form = button.closest("form");
        return form?.nextElementSibling?.getAttribute("role") === "alert";
      }),
    ).toBe(true);
  });

  test("gives empty Train and Prove states one product action", async ({
    page,
  }) => {
    await page.goto("/train?mode=no_training_yet");
    await expect(
      page.getByRole("link", { name: "Return to the project setup" }),
    ).toBeVisible();
    await expect(page.getByText("QUEUED")).not.toBeVisible();
    await expect(page.getByText("Not started", { exact: true })).toBeVisible();

    await page.goto("/prove?mode=default");
    await expect(page.getByRole("link", { name: "Set up a run" })).toBeVisible();
    await expect(page.getByText(/switch|fixture mode/i)).toHaveCount(0);
  });
});

test.describe("route stability", () => {
  for (const [route, heading] of ROUTES) {
    test(`${route} has one stable h1`, async ({ page }) => {
      await page.goto(route);
      await expect(
        page.getByRole("heading", { level: 1, name: heading }),
      ).toBeVisible();
      await expect(page.locator("h1")).toHaveCount(1);
    });
  }

  test("survives the walkthrough loop without browser errors", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });

    const durationMs = process.env.LONG_WALKTHROUGH === "1" ? 600_000 : 5_000;
    const dwellMs = process.env.LONG_WALKTHROUGH === "1" ? 10_000 : 0;
    test.setTimeout(durationMs + 60_000);
    const deadline = Date.now() + durationMs;
    do {
      for (const [route, heading] of ROUTES) {
        await page.goto(route);
        await expect(
          page.getByRole("heading", { level: 1, name: heading }),
        ).toBeVisible();
        if (dwellMs > 0) await page.waitForTimeout(dwellMs);
      }
    } while (Date.now() < deadline);

    expect(errors).toEqual([]);
  });
});
