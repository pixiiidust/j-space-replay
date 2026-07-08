/**
 * The trace clock. When a video element is available it is the source of truth
 * (we bind to timeupdate and drive play/pause/rate). When the video is
 * unavailable (fixture clips are gitignored) we run a requestAnimationFrame
 * timer so the timeline is still fully scrubbable. Either way `time` advances
 * and `seek`/`stepGroup` snap to frame-group boundaries.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import type { Trace } from "../trace/types";
import {
  groupIndexAtTime,
  snapToGroupStart,
  stepGroupIndex,
  traceDuration,
} from "../trace/selectors";

export interface Clock {
  time: number;
  playing: boolean;
  speed: number;
  groupIndex: number;
  duration: number;
  play(): void;
  pause(): void;
  toggle(): void;
  setSpeed(s: number): void;
  seek(t: number): void;
  seekGroup(idx: number): void;
  stepGroup(dir: number): void;
}

export function useClock(
  trace: Trace,
  videoRef: RefObject<HTMLVideoElement | null>,
  videoAvailable: boolean,
): Clock {
  const duration = traceDuration(trace);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeedState] = useState(1);
  const rafRef = useRef<number | null>(null);
  const lastTsRef = useRef<number | null>(null);

  const seek = useCallback(
    (t: number) => {
      const clamped = Math.max(0, Math.min(duration, t));
      setTime(clamped);
      const v = videoRef.current;
      if (videoAvailable && v) v.currentTime = clamped;
    },
    [duration, videoAvailable, videoRef],
  );

  const seekGroup = useCallback(
    (idx: number) => {
      const g = trace.frame_groups[idx];
      if (g) seek(g.time_start);
    },
    [seek, trace.frame_groups],
  );

  const stepGroup = useCallback(
    (dir: number) => {
      const cur = groupIndexAtTime(trace, time);
      seekGroup(stepGroupIndex(trace, cur, dir));
    },
    [seekGroup, time, trace],
  );

  const play = useCallback(() => setPlaying(true), []);
  const pause = useCallback(() => setPlaying(false), []);
  const toggle = useCallback(() => setPlaying((p) => !p), []);

  const setSpeed = useCallback(
    (s: number) => {
      setSpeedState(s);
      const v = videoRef.current;
      if (videoAvailable && v) v.playbackRate = s;
    },
    [videoAvailable, videoRef],
  );

  // Video-backed clock: subscribe to the element.
  useEffect(() => {
    const v = videoRef.current;
    if (!videoAvailable || !v) return;
    const onTime = () => setTime(v.currentTime);
    const onEnded = () => setPlaying(false);
    v.addEventListener("timeupdate", onTime);
    v.addEventListener("ended", onEnded);
    v.playbackRate = speed;
    return () => {
      v.removeEventListener("timeupdate", onTime);
      v.removeEventListener("ended", onEnded);
    };
  }, [videoAvailable, videoRef, speed]);

  useEffect(() => {
    const v = videoRef.current;
    if (!videoAvailable || !v) return;
    if (playing) {
      try {
        const p = v.play();
        if (p && typeof p.then === "function") p.catch(() => setPlaying(false));
      } catch {
        setPlaying(false);
      }
    } else {
      try {
        v.pause();
      } catch {
        /* no-op (e.g. jsdom) */
      }
    }
  }, [playing, videoAvailable, videoRef]);

  // Fallback rAF clock when there is no video element.
  useEffect(() => {
    if (videoAvailable) return;
    if (!playing) {
      lastTsRef.current = null;
      return;
    }
    const tick = (ts: number) => {
      const last = lastTsRef.current;
      lastTsRef.current = ts;
      if (last != null) {
        const dt = ((ts - last) / 1000) * speed;
        setTime((prev) => {
          const next = prev + dt;
          if (next >= duration) {
            setPlaying(false);
            return duration;
          }
          return next;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      lastTsRef.current = null;
    };
  }, [playing, speed, duration, videoAvailable]);

  return {
    time,
    playing,
    speed,
    groupIndex: groupIndexAtTime(trace, time),
    duration,
    play,
    pause,
    toggle,
    setSpeed,
    seek,
    seekGroup,
    stepGroup,
  };
}

export { snapToGroupStart };
