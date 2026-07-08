import { defineConfig } from "@playwright/test";

/**
 * Optional end-to-end smoke pass over the full loop, run against the Vite dev
 * server with the fake-api middleware (zero backend, zero GPU). Not part of
 * `npm test` (that's vitest). Run with: npm run e2e.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  use: {
    baseURL: "http://localhost:5173",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "npm run dev",
    url: "http://localhost:5173/library",
    reuseExistingServer: true,
    timeout: 30_000,
  },
});
