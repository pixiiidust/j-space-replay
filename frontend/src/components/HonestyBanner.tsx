import { HONESTY_BANNER } from "../constants";

/** SPEC-mandated honesty banner, rendered verbatim on the replay screen. */
export function HonestyBanner() {
  return (
    <div className="honesty" role="note">
      <b>Honesty:</b> {HONESTY_BANNER}
    </div>
  );
}
