import type { DerivedEvent } from "../trace/events";

interface Props {
  events: DerivedEvent[];
  onPick(ev: DerivedEvent): void;
}

/**
 * Event log — mechanically derived threshold/peak/drop events, phrased as
 * measurements (layer + time). Click seeks the clock to the event.
 */
export function EventLog({ events, onPick }: Props) {
  return (
    <div className="panel">
      <div className="panel-h">
        <span>Event Log · derived</span>
        <span className="muted">{events.length}</span>
      </div>
      <div className="panel-b" style={{ padding: 0, maxHeight: 220, overflowY: "auto" }}>
        {events.length === 0 && (
          <div className="axis-note" style={{ padding: "6px 8px" }}>
            no readouts crossed the threshold
          </div>
        )}
        {events.map((e, i) => (
          <div key={i} className="event" onClick={() => onPick(e)} title={`group g${e.group}`}>
            <span className={"k " + e.kind}>{e.kind}</span>
            <span>{e.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
