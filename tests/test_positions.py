"""Position map: token index -> visual(frame group, patch row/col) | text | generated."""

from jsr.positions import PositionMap, build_position_map

VIDEO_PAD = 100
VISION_START = 101
VISION_END = 102


def make_ids(n_groups: int, rows: int, cols: int, pre_text: int = 3, post_text: int = 4):
    ids = [1] * pre_text + [VISION_START]
    ids += [VIDEO_PAD] * (n_groups * rows * cols)
    ids += [VISION_END] + [2] * post_text
    return ids


def test_build_position_map_shapes():
    ids = make_ids(n_groups=2, rows=3, cols=4)
    pm = build_position_map(
        ids,
        video_grid_thw=(2, 6, 8),  # pre-merge patch grid; 2x2 merge -> 3x4 merged
        merge_size=2,
        video_pad_id=VIDEO_PAD,
    )
    assert isinstance(pm, PositionMap)
    assert pm.n_tokens == len(ids)
    assert len(pm.visual_indices) == 2 * 3 * 4
    assert pm.kind(0) == "text"
    first_vis = pm.visual_indices[0]
    assert pm.kind(first_vis) == "visual"
    assert pm.visual_pos(first_vis) == (0, 0, 0)  # (group, row, col)


def test_visual_ordering_is_group_then_row_major():
    ids = make_ids(n_groups=2, rows=2, cols=3)
    pm = build_position_map(ids, video_grid_thw=(2, 4, 6), merge_size=2, video_pad_id=VIDEO_PAD)
    seq = [pm.visual_pos(i) for i in pm.visual_indices]
    assert seq[0] == (0, 0, 0)
    assert seq[1] == (0, 0, 1)
    assert seq[3] == (0, 1, 0)  # wraps to next row after cols=3
    assert seq[6] == (1, 0, 0)  # second frame group starts after 2*3 tokens
    assert seq[-1] == (1, 1, 2)


def test_group_token_indices():
    ids = make_ids(n_groups=3, rows=2, cols=2)
    pm = build_position_map(ids, video_grid_thw=(3, 4, 4), merge_size=2, video_pad_id=VIDEO_PAD)
    for g in range(3):
        idxs = pm.group_token_indices(g)
        assert len(idxs) == 4
        assert all(pm.visual_pos(i)[0] == g for i in idxs)


def test_mismatched_pad_count_raises():
    ids = make_ids(n_groups=2, rows=3, cols=4)
    try:
        build_position_map(ids, video_grid_thw=(2, 6, 6), merge_size=2, video_pad_id=VIDEO_PAD)
    except ValueError as e:
        assert "video pad" in str(e)
    else:
        raise AssertionError("expected ValueError on grid/pad mismatch")
