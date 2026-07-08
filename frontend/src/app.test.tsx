/**
 * Guards two SPEC-locked facts and a render crash-smoke:
 *  - the honesty banner text is VERBATIM from SPEC.md,
 *  - the strength axis label is "readout strength" (never confidence/probability),
 *  - <App/> renders its initial (upload) screen with the banner present.
 */
import { describe, it, expect } from "vitest";
import { renderToString } from "react-dom/server";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { HONESTY_BANNER, STRENGTH_AXIS_LABEL } from "./constants";
import { App } from "./App";

const here = dirname(fileURLToPath(import.meta.url));
const specPath = resolve(here, "..", "..", "SPEC.md");

function bannerFromSpec(): string {
  const lines = readFileSync(specPath, "utf8").split("\n");
  // the banner is the blockquote under "## Honesty banner"
  const start = lines.findIndex((l) => l.startsWith("> Demo-quality"));
  const block: string[] = [];
  for (let i = start; i < lines.length && lines[i].startsWith(">"); i++) {
    block.push(lines[i].replace(/^>\s?/, "").trim());
  }
  return block.join(" ");
}

describe("SPEC-locked UI facts", () => {
  it("honesty banner is verbatim from SPEC.md", () => {
    expect(HONESTY_BANNER).toBe(bannerFromSpec());
  });

  it("strength axis label never says confidence/probability", () => {
    expect(STRENGTH_AXIS_LABEL).toBe("readout strength");
    expect(STRENGTH_AXIS_LABEL).not.toMatch(/confidence|probability/i);
  });

  it("App renders the upload screen with the honesty banner", () => {
    const html = renderToString(<App />);
    expect(html).toContain("Demo-quality interpretability");
    expect(html).toContain("run replay");
  });
});
