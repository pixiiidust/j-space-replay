/**
 * Guards two locked product facts and a render crash-smoke:
 *  - the honesty banner text is VERBATIM the locked wording below (it used to
 *    be sourced from SPEC.md; the spec doc was moved out of the public repo,
 *    so the test now carries the contract itself),
 *  - the strength axis label is "readout strength" (never confidence/probability),
 *  - <App/> renders its initial (upload) screen with the banner present.
 */
import { describe, it, expect } from "vitest";
import { renderToString } from "react-dom/server";
import { HONESTY_BANNER, STRENGTH_AXIS_LABEL } from "./constants";
import { App } from "./App";

// The locked banner wording. Changing this is a PRODUCT decision (honesty
// framing), not a copy edit — do not "fix" one side to match the other
// without that intent.
const LOCKED_BANNER =
  "Demo-quality interpretability. Lens readouts are noisy, single-token, and " +
  "unvalidated on vision-language models. The J-lens method was validated on " +
  "Claude text models only (Anthropic workspace paper); this tool extrapolates " +
  "it to a VLM. Not suitable for mechanistic claims.";

describe("locked UI facts", () => {
  it("honesty banner is verbatim the locked wording", () => {
    expect(HONESTY_BANNER).toBe(LOCKED_BANNER);
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
