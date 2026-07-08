/**
 * Trace schema-version guard (mirrors SCHEMA_VERSION in src/jsr/schema.py).
 *
 * The backend's validate_trace already refuses to WRITE a wrong-version trace;
 * this is the viewer-side counterpart so an older frontend served a newer trace
 * (or vice-versa) fails with a clear, visible message instead of rendering
 * garbage or throwing deep inside a canvas renderer.
 */
export const SUPPORTED_TRACE_SCHEMA = 1;

export class UnsupportedSchemaError extends Error {
  constructor(readonly got: unknown) {
    super(
      `This trace uses schema version ${String(got)}, but this viewer only ` +
        `supports version ${SUPPORTED_TRACE_SCHEMA}. Update J-Space-Replay ` +
        `(frontend and backend) to matching versions and recompute the trace.`,
    );
    this.name = "UnsupportedSchemaError";
  }
}

/** Throw UnsupportedSchemaError unless `trace.schema` is the supported version. */
export function assertSupportedSchema(trace: { schema?: unknown }): void {
  if (trace.schema !== SUPPORTED_TRACE_SCHEMA) {
    throw new UnsupportedSchemaError(trace.schema);
  }
}
