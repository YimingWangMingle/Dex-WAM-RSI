#!/usr/bin/env python3
"""变 FPS 训练：sampling_ranges 变倍率（stride_max / max_stride）读侧验证。

直接手工构造 video_stride 布局的 ranges sqlite（含/不含 max_stride 列），
stub repository（Pool 只用到 catalog_id），断言：

1. 同 (seed, epoch, index) 两次 generate 结果完全一致（确定性）；
2. stride_max=10 时长 episode：video 49 个、action 288 个；video 差值恒 = 6k、
   action 差值恒 = k（k ∈ [1,10] 整数）；max(video)/max(action) <= episode_len-1；
3. max_stride=3 的 episode k 永不超过 3；max_stride=1 的 episode 恒 k=1
   （video 差值恒 6，与旧行为等价）；
4. stride_max=1 的 pool 走旧路径：video 差值恒 6；
5. 防呆：stride_max>1 遇 consecutive 布局 / 缺 max_stride 列的 ranges 直接报错。

用法：python scripts/test_sampling_ranges_stride.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from thinker_wm.data_prepare.sampling_ranges import (  # noqa: E402
    MOTUS_SAMPLING_RANGES_V1,
    SamplingRangesError,
    SamplingRangesPool,
)

CATALOG_ID = "test_stride_catalog"
NUM_VIDEO_FRAMES = 48
RATIO = 6
SPAN = NUM_VIDEO_FRAMES * RATIO  # 288


class _StubRepository:
    """Pool 只读 catalog_id 做一致性校验。"""

    catalog_id = CATALOG_ID


def _write_ranges(
    path: Path,
    *,
    layout: str = "video_stride",
    with_max_stride: bool = True,
) -> None:
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE episode_ranges ("
            "ordinal INTEGER PRIMARY KEY, "
            "episode_id TEXT NOT NULL UNIQUE, "
            "ratio INTEGER NOT NULL, "
            "intervals TEXT NOT NULL, "
            "text_payload TEXT NOT NULL, "
            "episode_len INTEGER NOT NULL"
            + (", max_stride INTEGER NOT NULL" if with_max_stride else "")
            + ")"
        )
        metadata = {
            "format": MOTUS_SAMPLING_RANGES_V1,
            "catalog_id": CATALOG_ID,
            "dataset": "test_stride",
            "num_video_frames": str(NUM_VIDEO_FRAMES),
            "action_video_fps_ratio": str(RATIO),
            "layout": layout,
            "state_action_policy": "real",
            "episodes": "3",
            "split": "all",
            "split_seed": "20260713",
            "created_at": "2026-08-13T00:00:00+00:00",
        }
        if with_max_stride:
            metadata["max_stride"] = "10"
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            sorted(metadata.items()),
        )
        payload = json.dumps({"kind": "fixed", "indices": [0]})
        # (episode_id, intervals, episode_len, max_stride)
        episodes = [
            ("ep_long", [[0, 5000]], 20000, 10),
            ("ep_mid", [[0, 1000]], 5000, 3),
            ("ep_short", [[0, 100]], 500, 1),
        ]
        rows = []
        for ordinal, (episode_id, intervals, episode_len, max_stride) in enumerate(
            episodes
        ):
            row = (
                ordinal,
                episode_id,
                RATIO,
                json.dumps(intervals),
                payload,
                episode_len,
            )
            if with_max_stride:
                row = row + (max_stride,)
            rows.append(row)
        placeholder = "(?, ?, ?, ?, ?, ?" + (", ?" if with_max_stride else "") + ")"
        connection.executemany(
            "INSERT INTO episode_ranges (ordinal, episode_id, ratio, intervals, "
            "text_payload, episode_len"
            + (", max_stride" if with_max_stride else "")
            + f") VALUES {placeholder}",
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def _check_selection(selection: dict, episode_len: int) -> tuple[bool, str]:
    """校验 video_stride 变倍率窗口的索引结构，返回 (ok, 失败原因)。"""
    video = selection["video_frame_indices"]
    action = selection["action_indices"]
    state = selection["state_indices"]
    limit = episode_len - 1
    if len(video) != NUM_VIDEO_FRAMES + 1:
        return False, f"video 条数 {len(video)} != {NUM_VIDEO_FRAMES + 1}"
    if len(action) != SPAN:
        return False, f"action 条数 {len(action)} != {SPAN}"
    if state != [video[0]]:
        return False, f"state {state} != [video[0]]"
    if max(video) > limit or max(action) > limit:
        return False, f"索引越界: max(video)={max(video)} max(action)={max(action)} limit={limit}"
    video_diffs = {b - a for a, b in zip(video, video[1:])}
    if len(video_diffs) != 1:
        return False, f"video 差值不唯一: {sorted(video_diffs)}"
    video_step = video_diffs.pop()
    if video_step <= 0 or video_step % RATIO != 0:
        return False, f"video 差值 {video_step} 不是 {RATIO} 的正整数倍"
    k = video_step // RATIO
    action_diffs = {b - a for a, b in zip(action, action[1:])}
    if action_diffs != {k}:
        return False, f"action 差值 {sorted(action_diffs)} != {{k={k}}}"
    if action[0] - video[0] != k:
        return False, f"action[0]-s = {action[0] - video[0]} != k={k}"
    return True, ""


def _write_video_only_ranges(
    path: Path, *, layout: str = "video_stride", ratio: int = RATIO
) -> None:
    """纯视频源（state_action_policy=absent）的 ranges：无 state/action 时间线。"""
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE episode_ranges ("
            "ordinal INTEGER PRIMARY KEY, "
            "episode_id TEXT NOT NULL UNIQUE, "
            "ratio INTEGER NOT NULL, "
            "intervals TEXT NOT NULL, "
            "text_payload TEXT NOT NULL, "
            "episode_len INTEGER NOT NULL, "
            "max_stride INTEGER NOT NULL)"
        )
        metadata = {
            "format": MOTUS_SAMPLING_RANGES_V1,
            "catalog_id": CATALOG_ID,
            "dataset": "test_video_only",
            "num_video_frames": str(NUM_VIDEO_FRAMES),
            "action_video_fps_ratio": str(ratio),
            "layout": layout,
            "state_action_policy": "absent",
            "episodes": "2",
            "split": "all",
            "split_seed": "20260713",
            "max_stride": "10",
            "created_at": "2026-08-13T00:00:00+00:00",
        }
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            sorted(metadata.items()),
        )
        payload = json.dumps({"kind": "fixed", "indices": [0]})
        if ratio == 1:
            # openvid 语义（20260813 拍板 ratio 1）：49 帧窗口，max_stride =
            # 1 + (hi-lo)//48——vo1_mid（166 帧，区间 118 起点）k 上限 3。
            rows = [
                (0, "vo1_long", 1, json.dumps([[0, 5000]]), payload, 20000, 10),
                (1, "vo1_mid", 1, json.dumps([[0, 117]]), payload, 166, 3),
            ]
        else:
            rows = [
                (0, "vo_long", ratio, json.dumps([[0, 5000]]), payload, 20000, 10),
                (1, "vo_short", ratio, json.dumps([[0, 0]]), payload, 166, 1),
            ]
        connection.executemany(
            "INSERT INTO episode_ranges (ordinal, episode_id, ratio, intervals, "
            "text_payload, episode_len, max_stride) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def main() -> int:
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        ranges_path = Path(tmpdir) / "test.ranges.sqlite"
        _write_ranges(ranges_path)
        repository = _StubRepository()

        pool_var = SamplingRangesPool(ranges_path, repository, seed=0, stride_max=10)
        pool_base = SamplingRangesPool(ranges_path, repository, seed=0)
        # 各 episode 的虚拟 index 区间（width = 有效起点总数）
        # ep_long [0,5001) / ep_mid [5001,6002) / ep_short [6002,6103)
        cases = [
            ("ep_long", 0, 5001, 20000, 10),
            ("ep_mid", 5001, 6002, 5000, 3),
            ("ep_short", 6002, 6103, 500, 1),
        ]

        # 1. 确定性：同 (seed, epoch, index) 两次 generate 完全一致
        ok = True
        for epoch in (0, 1, 2):
            pool_var.set_epoch(epoch)
            for index in (0, 1234, 5000, 5001, 6002, 6102):
                first = pool_var.generate(index)
                second = pool_var.generate(index)
                if first != second:
                    ok = False
                    print(f"  不一致: epoch={epoch} index={index}")
        print(f"[{'PASS' if ok else 'FAIL'}] 确定性: 同 (seed,epoch,index) 两次 generate 一致")
        results.append(ok)

        # 2+3. 变倍率结构 / 越界 / max_stride clamp
        for name, index_lo, index_hi, episode_len, max_stride in cases:
            ok = True
            seen_k: set[int] = set()
            for epoch in range(4):
                pool_var.set_epoch(epoch)
                for index in range(index_lo, index_hi, 37):
                    selection = pool_var.generate(index)
                    assert selection["episode_id"] == name
                    good, reason = _check_selection(selection, episode_len)
                    if not good:
                        ok = False
                        print(f"  {name} epoch={epoch} index={index}: {reason}")
                        continue
                    k = (selection["video_frame_indices"][1]
                         - selection["video_frame_indices"][0]) // RATIO
                    seen_k.add(k)
                    if not 1 <= k <= max_stride:
                        ok = False
                        print(f"  {name}: k={k} 超出 [1,{max_stride}]")
            print(
                f"[{'PASS' if ok else 'FAIL'}] {name}: 变倍率结构/越界/"
                f"max_stride={max_stride} clamp（观测 k={sorted(seen_k)}）"
            )
            results.append(ok)

        # 4. stride_max==1 的 pool 走旧路径：video 差值恒 6
        ok = True
        for epoch in range(3):
            pool_base.set_epoch(epoch)
            for index in range(0, len(pool_base), 53):
                selection = pool_base.generate(index)
                video = selection["video_frame_indices"]
                action = selection["action_indices"]
                if len(video) != NUM_VIDEO_FRAMES + 1 or len(action) != SPAN:
                    ok = False
                    break
                if {b - a for a, b in zip(video, video[1:])} != {RATIO}:
                    ok = False
                    break
                if {b - a for a, b in zip(action, action[1:])} != {1}:
                    ok = False
                    break
        print(f"[{'PASS' if ok else 'FAIL'}] stride_max=1: 旧路径 video 差值恒 {RATIO}、action 差值恒 1")
        results.append(ok)

        # 5a. stride_max>1 + consecutive 布局 → 报错
        consecutive_path = Path(tmpdir) / "consecutive.ranges.sqlite"
        _write_ranges(consecutive_path, layout="consecutive")
        try:
            SamplingRangesPool(consecutive_path, repository, stride_max=10)
            ok = False
        except SamplingRangesError:
            ok = True
        print(f"[{'PASS' if ok else 'FAIL'}] 防呆: stride_max>1 + consecutive 布局报错")
        results.append(ok)

        # 5b. stride_max>1 + 缺 max_stride 列的旧 ranges → 报错；stride_max=1 可用
        legacy_path = Path(tmpdir) / "legacy.ranges.sqlite"
        _write_ranges(legacy_path, with_max_stride=False)
        try:
            SamplingRangesPool(legacy_path, repository, stride_max=10)
            ok1 = False
        except SamplingRangesError:
            ok1 = True
        try:
            legacy_pool = SamplingRangesPool(legacy_path, repository)
            legacy_pool.generate(0)
            ok2 = True
        except Exception:
            ok2 = False
        ok = ok1 and ok2
        print(f"[{'PASS' if ok else 'FAIL'}] 防呆: 缺 max_stride 列时 stride_max>1 报错、stride_max=1 旧行为可用")
        results.append(ok)

        # 6. 纯视频源（absent）+ video_stride + stride_max（20260813 起支持）：
        # state/action 索引为空、video 49 帧差值恒 = RATIO*k、不越界、确定性
        vo_path = Path(tmpdir) / "video_only.ranges.sqlite"
        _write_video_only_ranges(vo_path)
        vo_pool = SamplingRangesPool(vo_path, repository, seed=0, stride_max=10)
        # vo_long [0,5001) / vo_short [5001,5002)
        ok = True
        seen_k: set[int] = set()
        for epoch in range(4):
            vo_pool.set_epoch(epoch)
            for index in list(range(0, 5001, 41)) + [5001]:
                selection = vo_pool.generate(index)
                if selection["state_indices"] != [] or selection["action_indices"] != []:
                    ok = False
                    print(f"  absent 索引非空: index={index}")
                    continue
                video = selection["video_frame_indices"]
                name = selection["episode_id"]
                episode_len = 20000 if name == "vo_long" else 166
                if len(video) != NUM_VIDEO_FRAMES + 1:
                    ok = False
                    print(f"  {name}: video 条数 {len(video)} != {NUM_VIDEO_FRAMES + 1}")
                    continue
                if max(video) > episode_len - 1:
                    ok = False
                    print(f"  {name}: 越界 max(video)={max(video)} limit={episode_len - 1}")
                    continue
                if selection != vo_pool.generate(index):
                    ok = False
                    print(f"  {name}: 同 (epoch,index) 两次 generate 不一致")
                    continue
                diffs = {b - a for a, b in zip(video, video[1:]) if b > a}
                if name == "vo_long":
                    if len(diffs) != 1:
                        ok = False
                        print(f"  vo_long: 差值不唯一 {sorted(diffs)}")
                        continue
                    step = diffs.pop()
                    if step <= 0 or step % RATIO != 0:
                        ok = False
                        print(f"  vo_long: 差值 {step} 不是 {RATIO} 的正整数倍")
                        continue
                    k = step // RATIO
                    if not 1 <= k <= 10:
                        ok = False
                        print(f"  vo_long: k={k} 超出 [1,10]")
                    seen_k.add(k)
                else:
                    # vo_short（166 < 289）：s=0 + 末帧填充，k 恒 1；非零差值
                    # 除末段 clamp 尾巴（< RATIO 的唯一一段）外恒为 RATIO。
                    nonzero = [b - a for a, b in zip(video, video[1:]) if b > a]
                    head, tail = nonzero[:-1], nonzero[-1] if nonzero else RATIO
                    if set(head) != {RATIO} or not 0 < tail <= RATIO:
                        ok = False
                        print(f"  vo_short: 非填充段差值 head={sorted(set(head))} tail={tail}（期望 head={{{RATIO}}}）")
        print(
            f"[{'PASS' if ok else 'FAIL'}] 纯视频源 video_stride: absent 索引为空/"
            f"变倍率抽帧/短 episode 填充（vo_long 观测 k={sorted(seen_k)}）"
        )
        results.append(ok)

        # 7. 纯视频源 ratio=1（openvid 20260813 拍板语义）+ windows_per_episode
        # （libero 短 episode 修复：组宽补齐到 batch_size，1 起点 episode 也
        # 可组批；generate 起点由 RNG 在有效域抽取，不依赖 pick 序号）
        vo1_path = Path(tmpdir) / "video_only_ratio1.ranges.sqlite"
        _write_video_only_ranges(vo1_path, ratio=1)
        vo1_pool = SamplingRangesPool(vo1_path, repository, seed=0, stride_max=10)
        # vo1_long [0,5001) / vo1_mid [5001,5119)
        ok = True
        seen_k1: set[int] = set()
        for epoch in range(4):
            vo1_pool.set_epoch(epoch)
            for index in list(range(0, 5001, 43)) + list(range(5001, 5119, 13)):
                selection = vo1_pool.generate(index)
                if selection["state_indices"] != [] or selection["action_indices"] != []:
                    ok = False
                    print(f"  ratio1: absent 索引非空: index={index}")
                    continue
                video = selection["video_frame_indices"]
                name = selection["episode_id"]
                episode_len = 20000 if name == "vo1_long" else 166
                k_max = 10 if name == "vo1_long" else 3
                if len(video) != NUM_VIDEO_FRAMES + 1 or max(video) > episode_len - 1:
                    ok = False
                    print(f"  ratio1 {name}: 条数/越界异常 max={max(video)}")
                    continue
                if selection != vo1_pool.generate(index):
                    ok = False
                    print(f"  ratio1 {name}: 同 (epoch,index) 两次 generate 不一致")
                    continue
                diffs = {b - a for a, b in zip(video, video[1:]) if b > a}
                if len(diffs) != 1:
                    ok = False
                    print(f"  ratio1 {name}: 差值不唯一 {sorted(diffs)}")
                    continue
                k = diffs.pop()
                if not 1 <= k <= k_max:
                    ok = False
                    print(f"  ratio1 {name}: k={k} 超出 [1,{k_max}]")
                    continue
                # 窗口末端不越出有效区间（起点域按 k 收缩 ⇒ s+48k ≤ 165，
                # 即 max(video) ≤ episode_len-1，前面已断言；此处补充 k 与
                # 起点的一致性：s ≤ 117 - 48*(k-1)）
                if name == "vo1_mid" and video[0] > 117 - NUM_VIDEO_FRAMES * (k - 1):
                    ok = False
                    print(f"  ratio1 vo1_mid: 起点域未按 k 收缩 s={video[0]} k={k}")
                    continue
                if name == "vo1_long":
                    seen_k1.add(k)
        print(
            f"[{'PASS' if ok else 'FAIL'}] 纯视频源 ratio=1: 49 帧窗口/步长=k∈[1,max_stride]/"
            f"域收缩（vo1_long 观测 k={sorted(seen_k1)}）"
        )
        results.append(ok)

        # 8. windows_per_episode=16（libero 修复语义）：vo1_mid 有效起点 118
        # ≥16 不变；vo_short 型 1 起点 episode 组宽补齐到 16（len=2*16=32），
        # 所有 index 可 generate（起点域 randrange(total_starts)，不越界）
        wpe_pool = SamplingRangesPool(
            vo_path, repository, seed=0, stride_max=10, windows_per_episode=16
        )
        ok = len(wpe_pool) == 32
        groups = wpe_pool.episode_groups
        if sorted(len(g) for g in groups.values()) != [16, 16]:
            ok = False
            print(f"  wpe: 组宽异常 {[len(g) for g in groups.values()]}")
        for epoch in range(2):
            wpe_pool.set_epoch(epoch)
            for index in range(32):
                selection = wpe_pool.generate(index)
                video = selection["video_frame_indices"]
                name = selection["episode_id"]
                episode_len = 20000 if name == "vo_long" else 166
                if len(video) != NUM_VIDEO_FRAMES + 1 or max(video) > episode_len - 1:
                    ok = False
                    print(f"  wpe {name}: index={index} 条数/越界异常")
                    break
                if selection != wpe_pool.generate(index):
                    ok = False
                    print(f"  wpe {name}: index={index} 两次 generate 不一致")
                    break
        print(
            f"[{'PASS' if ok else 'FAIL'}] windows_per_episode=16: 组宽补齐/"
            f"1 起点 episode 全 index 可生成/确定性"
        )
        results.append(ok)

    print("=" * 80)
    print(f"SAMPLING_RANGES_STRIDE_TEST {'ALL_PASS' if all(results) else 'FAILED'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
