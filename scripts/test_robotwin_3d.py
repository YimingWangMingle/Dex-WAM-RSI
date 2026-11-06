#!/usr/bin/env python3
"""Unit tests for Motus T compose / shared-scale depth pad."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from thinker_wm.data_prepare.robotwin_3d import (  # noqa: E402
    COMPOSE_HW,
    compose_motus_t_depth_mm,
    compose_motus_t_rgb,
    pad_depth_mm_to_size,
    pad_rgb_to_size,
    parse_episode_index,
)
from thinker_wm.models.action_stats import denormalize, get_stats, normalize  # noqa: E402
from thinker_wm.models.depth_codec import encode_depth_mm  # noqa: E402


def test_parse_id() -> None:
    assert parse_episode_index("robotwin_3d:000012") == 12


def test_compose_layout() -> None:
    head = np.full((240, 320, 3), 10, dtype=np.uint8)
    left = np.full((240, 320, 3), 80, dtype=np.uint8)
    right = np.full((240, 320, 3), 160, dtype=np.uint8)
    canvas = compose_motus_t_rgb(head, left, right)
    assert canvas.shape == (360, 320, 3)
    assert canvas[0, 0, 0] == 10
    assert canvas[240, 0, 0] == 80
    assert canvas[240, 160, 0] == 160
    # wrist downsample should still be a solid fill
    assert int(canvas[300, 80, 0]) == 80
    assert int(canvas[300, 240, 0]) == 160


def test_depth_keeps_metric_across_views() -> None:
    head = np.full((240, 320), 700, dtype=np.uint16)
    left = np.full((240, 320), 90, dtype=np.uint16)
    right = np.full((240, 320), 50, dtype=np.uint16)
    canvas = compose_motus_t_depth_mm(head, left, right)
    assert canvas.shape == COMPOSE_HW
    assert canvas[120, 160] == 700
    assert canvas[300, 80] == 90
    assert canvas[300, 240] == 50


def test_shared_log_keeps_wrist_nearer_than_head() -> None:
    canvas = compose_motus_t_depth_mm(
        np.full((240, 320), 700, dtype=np.uint16),
        np.full((240, 320), 90, dtype=np.uint16),
        np.full((240, 320), 50, dtype=np.uint16),
    )
    encoded = encode_depth_mm(
        torch.from_numpy(canvas.astype(np.int32)),
        z_near=0.01,
        z_far=2.0,
        mode="log",
        invalid_to="far",
        out_range="01",
    )
    # near=bright: 50 mm wrist > 90 mm wrist > 700 mm head
    v_head = float(encoded[0, 120, 160])
    v_left = float(encoded[0, 300, 80])
    v_right = float(encoded[0, 300, 240])
    assert v_right > v_left > v_head > 0.0


def test_pad_alignment() -> None:
    rgb = np.zeros((1, 360, 320, 3), dtype=np.uint8)
    rgb[0, 0, 0] = 255
    depth = np.zeros((1, 360, 320), dtype=np.uint16)
    depth[0, 0, 0] = 500
    rgb_p = pad_rgb_to_size(rgb, (480, 640))
    dep_p = pad_depth_mm_to_size(depth, (480, 640))
    assert rgb_p.shape == (1, 480, 640, 3)
    assert dep_p.shape == (1, 480, 640)
    ys, xs = np.where(dep_p[0] > 0)
    assert ys.size >= 1
    # same content origin after shared pad
    assert rgb_p[0, int(ys.min()), int(xs.min()), 0] == 255
    assert dep_p[0, int(ys.min()), int(xs.min())] == 500
    # side bars stay invalid 0
    assert int(dep_p[0, 240, 0]) == 0


def test_aloha_zscore_roundtrip() -> None:
    stats = get_stats("aloha")
    assert stats is not None
    mean, std = stats
    assert tuple(mean.shape) == (14,)
    assert torch.all(std > 0)
    raw = mean.clone()
    z = normalize(raw, "aloha")
    assert torch.allclose(z, torch.zeros(14), atol=1e-5)
    back = denormalize(z, "aloha")
    assert torch.allclose(back, raw, atol=1e-5)
    assert get_stats("not-a-robot") is None


def main() -> int:
    test_parse_id()
    test_compose_layout()
    test_depth_keeps_metric_across_views()
    test_shared_log_keeps_wrist_nearer_than_head()
    test_pad_alignment()
    test_aloha_zscore_roundtrip()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
