"""Video frame sampling via PyAV.

We decode with PyAV ourselves instead of qwen-vl-utils' built-in readers:
torchvision >= 0.24 removed `io.read_video` (breaking that backend), decord is
banned (unmaintained, flaky on Windows), and torchcodec needs system FFmpeg
DLLs. PyAV bundles FFmpeg in its wheel — nothing to install, nothing compiled.

Sampled frames are handed to qwen-vl-utils as a list-of-frames video element,
which handles smart resizing (max_pixels) and even-frame padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
from PIL import Image


@dataclass
class SampledVideo:
    frames: list[Image.Image]
    timestamps: list[float]  # seconds, one per sampled frame
    duration: float  # container duration in seconds
    native_fps: float

    @property
    def sample_fps(self) -> float:
        if len(self.timestamps) < 2:
            return 1.0
        return (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])


def sample_frames(path: str | Path, fps: float = 1.0) -> SampledVideo:
    """Decode `path` and keep roughly `fps` frames per second.

    Keeps the first frame at/after each 1/fps boundary. Decoding a whole
    <= 25 s clip is cheap; no seeking needed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frames: list[Image.Image] = []
    timestamps: list[float] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        native_fps = float(stream.average_rate) if stream.average_rate else 0.0
        duration = (
            float(container.duration / av.time_base) if container.duration else 0.0
        )
        step = 1.0 / fps
        next_t = 0.0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            if t + 1e-6 >= next_t:
                frames.append(frame.to_image())
                timestamps.append(t)
                next_t += step
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return SampledVideo(frames, timestamps, duration or timestamps[-1], native_fps)
