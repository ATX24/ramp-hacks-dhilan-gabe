import { test, expect } from "@playwright/test";

const RESTORED_RUN_ID = "run_fixture_tinyfable_restored_002";

test.describe("five-stage Distillery UI", () => {
  test("root redirects to Curate", async ({ page }) => {
    await page.goto("/?mode=failed_quality");
    await expect(page).toHaveURL(/\/curate\?mode=failed_quality/);
    await expect(page.getByRole("heading", { name: "Curate" })).toBeVisible();
  });

  test("renders exactly five routes", async ({ page }) => {
    for (const [route, heading] of [
      ["/curate", "Curate"],
      ["/synthesize", "Synthesize"],
      ["/train", "Train"],
      ["/prove", "Prove"],
      ["/demo", "Demo"],
    ] as const) {
      await page.goto(`${route}?mode=default`);
      await expect(page.getByRole("heading", { name: heading })).toBeVisible();
    }
  });

  test("preserves mode and run through stage navigation", async ({ page }) => {
    await page.goto("/curate?mode=failed_quality");
    await expect(page.getByTestId("run-reference")).toHaveAttribute(
      "data-status",
      "stored",
    );
    const synthesize = page.getByTestId("stage-link-synthesize");
    await expect(synthesize).toHaveAttribute(
      "href",
      /\/synthesize\?mode=failed_quality&run=run_fixture_failed_quality_001/,
    );
    await synthesize.click();
    await expect(page).toHaveURL(
      /\/synthesize\?mode=failed_quality&run=run_fixture_failed_quality_001/,
    );
  });

  test("reconstructs stored run on deep navigation and refresh", async ({ page }) => {
    await page.goto(`/curate?mode=default&run=${RESTORED_RUN_ID}`);
    await expect(page.getByTestId("run-reference")).toContainText(RESTORED_RUN_ID);

    await page.goto("/train?mode=default");
    await expect(page.getByTestId("run-reference")).toContainText(RESTORED_RUN_ID);
    await page.reload();
    await expect(page.getByTestId("run-reference")).toContainText(RESTORED_RUN_ID);
  });

  test("remains usable at a narrow viewport", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/prove?mode=proved");
    await expect(page.getByRole("heading", { name: "Prove" })).toBeVisible();
    await expect(page.getByTestId("stage-link-curate")).toBeVisible();
    const overflow = await page.evaluate(() =>
      Array.from(document.querySelectorAll<HTMLElement>("body *"))
        .filter((element) => {
          if (
            element.closest(".table-wrap") &&
            !element.classList.contains("table-wrap")
          ) {
            return false;
          }
          const rect = element.getBoundingClientRect();
          return rect.right > window.innerWidth + 1 || rect.left < -1;
        })
        .map((element) => ({
          tag: element.tagName,
          className: element.className,
          text: element.textContent?.trim().slice(0, 80),
          right: element.getBoundingClientRect().right,
        })),
    );
    expect(overflow).toEqual([]);
  });

  test("supports keyboard activation of the Curate continuation", async ({ page }) => {
    await page.goto("/curate?mode=default");
    const continueLink = page.getByTestId("curate-continue");
    await continueLink.focus();
    await expect(continueLink).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/\/synthesize\?mode=default&run=/);
  });
});
