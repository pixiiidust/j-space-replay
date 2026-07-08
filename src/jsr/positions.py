"""Position map: token index -> {visual: (frame group, patch row, col)} | text | generated.

Qwen2.5-VL packs video into the prompt as a run of video-pad tokens ordered
temporal-group-major, then row-major over the *merged* patch grid:
`video_grid_thw = (T, H, W)` counts 14 px patches before the 2x2 spatial merge,
so the token grid is T x (H/merge) x (W/merge) and temporal group t corresponds
to frame group t (2-frame temporal merge).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PositionMap:
    n_tokens: int
    visual_indices: list[int]  # token indices that are video-pad positions, in order
    grid: tuple[int, int, int]  # (groups, merged rows, merged cols)
    _vis_order: dict[int, int] = field(repr=False, default_factory=dict)

    def kind(self, token_index: int) -> str:
        if token_index >= self.n_tokens:
            return "generated"
        return "visual" if token_index in self._vis_order else "text"

    def visual_pos(self, token_index: int) -> tuple[int, int, int]:
        """(frame group, patch row, patch col) for a visual token index."""
        k = self._vis_order[token_index]
        t, rows, cols = self.grid
        per_group = rows * cols
        return (k // per_group, (k % per_group) // cols, k % cols)

    def group_token_indices(self, group: int) -> list[int]:
        t, rows, cols = self.grid
        per_group = rows * cols
        return self.visual_indices[group * per_group : (group + 1) * per_group]


def build_position_map(
    input_ids: list[int],
    video_grid_thw: tuple[int, int, int],
    merge_size: int,
    video_pad_id: int,
) -> PositionMap:
    t, h, w = video_grid_thw
    rows, cols = h // merge_size, w // merge_size
    visual_indices = [i for i, tok in enumerate(input_ids) if tok == video_pad_id]
    expected = t * rows * cols
    if len(visual_indices) != expected:
        raise ValueError(
            f"video pad count {len(visual_indices)} != grid {t}x{rows}x{cols} = {expected}"
        )
    return PositionMap(
        n_tokens=len(input_ids),
        visual_indices=visual_indices,
        grid=(t, rows, cols),
        _vis_order={idx: k for k, idx in enumerate(visual_indices)},
    )
