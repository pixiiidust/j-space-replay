import { describe, it, expect } from "vitest";
import {
  SUPPORTED_TRACE_SCHEMA,
  UnsupportedSchemaError,
  assertSupportedSchema,
} from "./schema";

describe("trace schema guard", () => {
  it("accepts the supported version", () => {
    expect(() => assertSupportedSchema({ schema: SUPPORTED_TRACE_SCHEMA })).not.toThrow();
  });

  it("rejects a newer/unknown version with a clear message", () => {
    expect(() => assertSupportedSchema({ schema: 2 })).toThrow(UnsupportedSchemaError);
    try {
      assertSupportedSchema({ schema: 2 });
    } catch (e) {
      expect(String(e)).toMatch(/schema version 2/);
      expect(String(e)).toMatch(/only supports version 1/);
    }
  });

  it("rejects a missing version", () => {
    expect(() => assertSupportedSchema({})).toThrow(UnsupportedSchemaError);
  });
});
