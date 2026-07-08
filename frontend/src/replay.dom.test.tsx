/**
 * jsdom mount smoke for the full replay dashboard. jsdom has no canvas 2D
 * context, so the canvas renderers no-op (they guard on a null ctx) — but every
 * hook, effect and DOM panel is exercised, which catches runtime crashes that a
 * server-render can't. Uses a real committed fixture trace via a mocked fetch.
 *
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeAll, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { render, screen, fireEvent, waitFor, cleanup } from "@testing-library/react";
import { ReplayScreen } from "./screens/ReplayScreen";

const here = dirname(fileURLToPath(import.meta.url));
const tracePath = resolve(here, "..", "..", "fixtures", "traces", "ball_drop.trace.json");
const trace = JSON.parse(readFileSync(tracePath, "utf8"));

beforeAll(() => {
  // deterministic clock/raf
  globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) =>
    setTimeout(() => cb(performance.now()), 16) as unknown as number) as typeof requestAnimationFrame;
  globalThis.cancelAnimationFrame = ((id: number) => clearTimeout(id)) as typeof cancelAnimationFrame;
  // fetch returns the fixture trace for GET /traces/:id
  globalThis.fetch = (async (url: string | URL) => {
    const u = String(url);
    if (u.includes("/traces/")) {
      return new Response(JSON.stringify(trace), { status: 200 });
    }
    return new Response("{}", { status: 404 });
  }) as typeof fetch;
});

afterEach(() => cleanup());

describe("replay dashboard mounts and drives", () => {
  it("renders the dashboard, honesty banner, timeline, concepts, events", async () => {
    render(<ReplayScreen traceId="ball_drop" onReAsk={() => {}} onOpenLibrary={() => {}} />);

    // banner (verbatim marker) and core panels appear once the trace loads
    await waitFor(() => expect(screen.getByText(/Demo-quality interpretability/)).toBeTruthy());
    expect(screen.getByText(/J-Space Timeline/)).toBeTruthy();
    expect(screen.getByText(/Concepts/)).toBeTruthy();
    expect(screen.getByText(/Event Log/)).toBeTruthy();
    // fixtures have no concept labels -> fallback banner
    expect(screen.getAllByText(/no concept labels in this trace/).length).toBeGreaterThan(0);
    // strength axis label present, never "confidence"
    expect(screen.getAllByText(/readout strength/).length).toBeGreaterThan(0);
    expect(document.body.textContent).not.toMatch(/confidence|probability/i);
  });

  it("supports playback controls and re-ask without crashing", async () => {
    let reAsked = false;
    render(
      <ReplayScreen
        traceId="ball_drop"
        onReAsk={() => (reAsked = true)}
        onOpenLibrary={() => {}}
      />,
    );
    await waitFor(() => screen.getByText(/▶ play/));

    fireEvent.click(screen.getByText(/▶ play/));
    fireEvent.click(screen.getByText(/step ▶/));
    fireEvent.click(screen.getByText("re-ask ▶"));
    expect(reAsked).toBe(true);
  });
});
