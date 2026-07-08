import type { Clock } from "../hooks/useClock";
import type { TimelineModel } from "../trace/selectors";
import { activeRowsAtGroup } from "../trace/selectors";
import { PLAYBACK_SPEEDS, STRENGTH_AXIS_LABEL } from "../constants";
import { fmtClock } from "../format";
import type { Trace } from "../trace/types";

interface Props {
  clock: Clock;
  trace: Trace;
  model: TimelineModel;
}

/** Playback controls (play/pause, speeds, step group, scrubber snapping to
 *  group boundaries) plus the live "active readouts" HUD. */
export function Controls({ clock, trace, model }: Props) {
  const active = activeRowsAtGroup(model, clock.groupIndex, 5);
  const maxActive = active.reduce((m, a) => Math.max(m, a.strength), 0) || model.maxStrength;

  return (
    <div className="panel">
      <div className="panel-h">
        <span>Transport</span>
        <span className="muted">
          group g{clock.groupIndex} / {trace.frame_groups.length}
        </span>
      </div>
      <div className="panel-b">
        <div className="clock tnum">{fmtClock(clock.time)}</div>

        <div className="controls" style={{ marginBottom: 4 }}>
          <button className="btn" onClick={clock.toggle}>
            {clock.playing ? "❚❚ pause" : "▶ play"}
          </button>
          <button className="btn" title="previous frame group" onClick={() => clock.stepGroup(-1)}>
            ◀ step
          </button>
          <button className="btn" title="next frame group" onClick={() => clock.stepGroup(+1)}>
            step ▶
          </button>
        </div>

        <div className="controls" style={{ marginBottom: 6 }}>
          {PLAYBACK_SPEEDS.map((s) => (
            <button
              key={s}
              className={"btn" + (clock.speed === s ? " active" : "")}
              onClick={() => clock.setSpeed(s)}
            >
              {s}x
            </button>
          ))}
        </div>

        {/* free scrubber */}
        <input
          className="scrubber"
          type="range"
          min={0}
          max={clock.duration}
          step={0.01}
          value={clock.time}
          onChange={(e) => clock.seek(parseFloat(e.target.value))}
          aria-label="scrubber"
        />
        {/* frame-group snapping rail */}
        <div className="grouprail" title="frame groups (click to snap)">
          {trace.frame_groups.map((g, i) => (
            <div
              key={g.group}
              className={"seg" + (i === clock.groupIndex ? " on" : "")}
              title={`g${g.group} ${g.time_start}-${g.time_end}s`}
              onClick={() => clock.seekGroup(i)}
            />
          ))}
        </div>

        {/* active readouts HUD */}
        <div className="hud" style={{ marginTop: 8 }}>
          <div className="axis-note" style={{ marginBottom: 3 }}>
            active ({STRENGTH_AXIS_LABEL}):
          </div>
          {active.length === 0 && <div className="muted">— no active readouts —</div>}
          {active.map((a) => (
            <div className="row" key={a.label}>
              <div className="lbl">{a.label}</div>
              <div className="bar">
                <span style={{ width: `${Math.min(100, (a.strength / maxActive) * 100)}%` }} />
              </div>
              <div className="tnum">{a.strength.toFixed(2)}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
