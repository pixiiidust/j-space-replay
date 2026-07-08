import { test, expect } from "@playwright/test";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const anyFile = resolve(here, "..", "package.json"); // upload accepts any file in dev

/**
 * Full loop with no dev-tools knowledge:
 * upload → progress → replay → scrub → drill into a cell → re-ask.
 */
test("full replay loop", async ({ page }) => {
  await page.goto("/");

  // 1. upload + question (default prefilled) + run
  await expect(page.getByRole("heading", { name: /video j-space replay/i })).toBeVisible();
  await page.locator('input[type="file"]').setInputFiles(anyFile);
  await expect(page.getByText(/selected:/)).toBeVisible();
  await page.getByRole("button", { name: /run replay/i }).click();

  // 2. progress → auto-transition to replay
  await expect(page.getByText(/Demo-quality interpretability/)).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText(/J-Space Timeline/)).toBeVisible();
  await page.screenshot({ path: "e2e/__screens__/replay.png", fullPage: true });

  // 3. scrub: click the group rail and step
  await page.getByRole("button", { name: /step ▶/ }).click();
  await page.getByRole("button", { name: /0.5x/ }).click();

  // 4. drill into a timeline cell (canvas click) → j-space raw tokens
  //    (exclude the video overlay canvas; the timeline is the first data canvas)
  const canvas = page.locator("canvas:not(.video-overlay)").first();
  await canvas.click({ position: { x: 160, y: 40 } });
  await expect(page.getByText(/raw top-10/)).toBeVisible();

  // pin a concept + jump to peak
  await page.getByRole("button", { name: /^peak$/ }).first().click();
  await page.getByRole("button", { name: /^pin$/ }).first().click();

  // 5. re-ask → back through the pipeline → replay again
  await page.getByRole("textbox", { name: /new question/i }).fill("What color is the ball?");
  await page.getByRole("button", { name: /re-ask/i }).click();
  await expect(page.getByText(/Demo-quality interpretability/)).toBeVisible({ timeout: 15_000 });
  await page.screenshot({ path: "e2e/__screens__/reask.png", fullPage: true });

  // library screen
  await page.getByRole("button", { name: /^library$/ }).click();
  await expect(page.getByText(/^Library$/)).toBeVisible();
  await page.screenshot({ path: "e2e/__screens__/library.png", fullPage: true });
});
