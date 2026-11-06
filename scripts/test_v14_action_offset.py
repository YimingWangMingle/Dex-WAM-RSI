#!/usr/bin/env python3
"""CPU regression: V14 action[t] follows state[t]; legacy defaults unchanged."""
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from test_sampling_ranges_stride import _write_ranges, _StubRepository
from thinker_wm.data_prepare.sampling_ranges import (
    SamplingRangesPool,
    _refine_intervals_by_outliers,
    build_selection,
)


class _SelectionEpisode:
    episode_id = "offset-selection"
    info = SimpleNamespace(
        video_count=100, state_count=100, action_count=100,
        source_dataset="robotwin_3d",
    )
    text = SimpleNamespace(items=[SimpleNamespace(
        index=0, kind="instruction_en", text="move", frame_range=None
    )])

    @staticmethod
    def _validated_selection_indices(selection):
        return None


def _check_build_selection_and_filter():
    episode = _SelectionEpisode()
    for layout, state_index in (("video_stride", 3), ("consecutive", 6)):
        new = build_selection(
            episode, start=3, num_video_frames=2, ratio=2,
            layout=layout, action_start_offset=0,
        )
        old = build_selection(
            episode, start=3, num_video_frames=2, ratio=2,
            layout=layout, action_start_offset=1,
        )
        assert new["state_indices"] == old["state_indices"] == [state_index]
        assert new["action_indices"] == list(range(state_index, state_index + 4))
        assert old["action_indices"] == list(range(state_index + 1, state_index + 5))

    # offset=1 的窗口 [1,2] 命中 index 2 离群值；offset=0 的 [0,1] 不命中。
    action = torch.zeros(6, 1)
    action[2] = 100
    kept, rejected = _refine_intervals_by_outliers(
        [(0, 0)], action, 1, 2, 10, layout="video_stride",
        action_start_offset=0,
    )
    assert kept == [(0, 0)] and rejected == 0
    kept, rejected = _refine_intervals_by_outliers(
        [(0, 0)], action, 1, 2, 10, layout="video_stride",
        action_start_offset=1,
    )
    assert kept == [] and rejected == 1
    print("PASS build_selection/outlier filter honor action_start_offset")


def main():
    _check_build_selection_and_filter()
    with tempfile.TemporaryDirectory() as tmp:
        for layout, stride_max, short in (
            ("video_stride", 1, False),
            ("video_stride", 3, False),
            ("video_stride", 1, True),
            ("consecutive", 1, False),
        ):
            path = Path(tmp) / f"{layout}_{stride_max}_{short}.sqlite"
            _write_ranges(path, layout=layout)
            with sqlite3.connect(path) as db:
                db.execute("UPDATE metadata SET value='24' WHERE key='num_video_frames'")
                db.execute("UPDATE metadata SET value='2' WHERE key='action_video_fps_ratio'")
                db.execute("DELETE FROM episode_ranges WHERE ordinal != 0")
                db.execute("UPDATE episode_ranges SET ratio=2, intervals=?, episode_len=?, max_stride=?",
                           ('[[0,0]]' if short else '[[0,100]]', 17 if short else 1000, stride_max))
            pools = [SamplingRangesPool(path, _StubRepository(), seed=19,
                                        stride_max=stride_max, **kw)
                     for kw in ({}, {"action_start_offset": 1}, {"action_start_offset": 0})]
            seen_k = set()
            for epoch in (0, 1, 2):
                for pool in pools:
                    pool.set_epoch(epoch)
                for i in range(len(pools[0])):
                    default, old, new = [p.generate(i) for p in pools]
                    assert default == old
                    assert new == pools[2].generate(i)
                    for key in ("video_frame_indices", "state_indices", "text_indices"):
                        assert old[key] == new[key], key
                    s = new["state_indices"][0]
                    k = ((new["video_frame_indices"][1] - new["video_frame_indices"][0]) // 2
                         if layout == "video_stride" else 1)
                    seen_k.add(k)
                    want = [min(s + k*j, 16) if short else s+k*j for j in range(48)]
                    assert new["action_indices"] == want
                    assert new["action_indices"][0] == s
                    # Toy episode: action[t] stores state[t+1]. The first command
                    # after observing state[s] must lead to state[s+1], not s+2.
                    assert new["action_indices"][0] + 1 == s + 1
                    if short:
                        assert new["action_indices"][-1] == 16
            if stride_max > 1:
                assert seen_k == {1, 2, 3}, seen_k
            print(f"PASS {layout} stride_max={stride_max} short={short}")
    print("All offset=0/legacy/determinism/window tests passed")


if __name__ == "__main__":
    main()
