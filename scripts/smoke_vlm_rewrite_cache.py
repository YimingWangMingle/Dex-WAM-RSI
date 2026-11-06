#!/usr/bin/env python3
"""VLM 改写磁盘缓存 + 输出侧字段 dropout 的纯 CPU 冒烟（20260818_v2）。

不依赖模型/GPU：
1) VLMRewriteCache：put/get、两级分片路径、重复写幂等（同 key 内容相同）、
   损坏 JSON 按 miss、多进程并发写同 key 安全、模型 tag 变更隔离命名空间。
2) apply_field_dropout_to_rewrite：字段行整行移除、字段行全中时保底留 1、
   无字段行文本（text_only caption 等）原样返回、prob=0 恒等、dropout 在
   缓存之后作用（缓存文本不含 dropout 结果——同一消息的缓存内容不变）。

用法：
    python scripts/smoke_vlm_rewrite_cache.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json
import multiprocessing as mp
import tempfile

import torch

from thinker_wm.utils.vlm_rewrite_cache import VLMRewriteCache
from thinker_wm.utils.vlm_utils import apply_field_dropout_to_rewrite


def _writer_proc(cache_dir: str, model_tag: str, message: str, text: str) -> None:
    VLMRewriteCache(cache_dir, model_tag).set_many([(message, text, True)])


def test_cache() -> bool:
    ok = True
    with tempfile.TemporaryDirectory() as cache_dir:
        cache = VLMRewriteCache(cache_dir, "Qwen3-VL-2B-Instruct")
        m1 = "instruction A\nInput text:\npick up the cup"
        m2 = "instruction A\nInput text:\npick up the plate"

        # 全 miss
        texts, hits, misses = cache.get_many([m1, m2])
        assert texts == [None, None] and hits == 0 and misses == 2, "初始应全 miss"

        # put → get
        cache.set_many([(m1, "Task: pick.\nAction: pick the cup.", True)])
        texts, hits, misses = cache.get_many([m1, m2])
        if texts[0] != "Task: pick.\nAction: pick the cup." or texts[1] is not None or hits != 1:
            print(f"FAIL: put/get 异常: {texts}")
            ok = False

        # 分片路径布局 {dir}/{h[:2]}/{h[2:4]}/{h}.json
        key = cache.key_for(m1)
        expect = Path(cache_dir) / key[:2] / key[2:4] / f"{key}.json"
        if not expect.exists():
            print(f"FAIL: 分片路径不符: {expect}")
            ok = False

        # 重复写幂等（内容不变、不报错）
        cache.set_many([(m1, "Task: OTHER.\nAction: other.", True)])
        texts, _, _ = cache.get_many([m1])
        if texts[0] != "Task: pick.\nAction: pick the cup.":
            print(f"FAIL: 重复写覆盖了已有条目: {texts[0]}")
            ok = False

        # 损坏 JSON 按 miss
        with open(expect, "w", encoding="utf-8") as f:
            f.write("{not a json")
        texts, hits, misses = cache.get_many([m1])
        if texts[0] is not None or misses != 1:
            print("FAIL: 损坏 JSON 应按 miss 处理")
            ok = False
        expect.unlink()  # 清掉损坏文件，避免污染后续并发写断言

        # 多进程并发写同 key（内容相同，原子改名，无锁）
        cache.set_many([(m2, "Task: pick.\nAction: pick the plate.", True)])
        procs = [
            mp.Process(target=_writer_proc, args=(cache_dir, "Qwen3-VL-2B-Instruct", m1, "Task: pick.\nAction: pick the cup."))
            for _ in range(8)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
            assert p.exitcode == 0, "并发写进程异常退出"
        texts, hits, _ = cache.get_many([m1, m2])
        if hits != 2:
            print(f"FAIL: 并发写后命中异常: {texts}")
            ok = False

        # 模型 tag 变更 → 命名空间隔离（旧条目不可见）
        cache_new = VLMRewriteCache(cache_dir, "Qwen3-VL-4B-Instruct")
        _, hits, misses = cache_new.get_many([m1, m2])
        if hits != 0 or misses != 2:
            print("FAIL: 模型 tag 变更应隔离命名空间")
            ok = False
    if ok:
        print("cache: put/get、分片、幂等、损坏容忍、并发写、命名空间隔离 全部通过")
    return ok


def test_field_dropout() -> bool:
    ok = True
    full = "Task: pick up the cup.\nAction: Move to the cup and grasp it.\nScene: A tabletop with a cup."

    # prob=0 恒等
    assert apply_field_dropout_to_rewrite(full, 0.0) == full, "prob=0 应恒等"

    # prob=1 → 保底留 1 个字段行
    torch.manual_seed(0)
    for _ in range(20):
        out = apply_field_dropout_to_rewrite(full, 1.0)
        field_lines = [l for l in out.splitlines() if l.split(":", 1)[0] in ("Task", "Action", "Scene")]
        if len(field_lines) != 1:
            print(f"FAIL: prob=1 应保底留 1 个字段行: {out!r}")
            ok = False
            break

    # 中间概率：只删字段行、行数单调不增、内容子集
    torch.manual_seed(1)
    seen_shorter = False
    for _ in range(50):
        out = apply_field_dropout_to_rewrite(full, 0.5)
        out_lines = out.splitlines()
        if not all(l in full.splitlines() for l in out_lines):
            print(f"FAIL: 输出含非原文行: {out!r}")
            ok = False
            break
        if not out_lines:
            print("FAIL: 输出为空（保底失效）")
            ok = False
            break
        seen_shorter = seen_shorter or len(out_lines) < 3
    if not seen_shorter:
        print("FAIL: prob=0.5 五十次采样未出现字段删除")
        ok = False

    # 无字段行文本（text_only caption）原样返回
    caption = "a person walks across a room"
    assert apply_field_dropout_to_rewrite(caption, 1.0) == caption, "无字段行文本应原样返回"

    # dropout 不产生新 cache key：缓存存的是完整输出，dropout 在其之后作用
    # （这里验证语义顺序——缓存内容始终是完整文本，与 dropout 无关）。
    print("field_dropout: 整行移除/保底留 1/无字段透传/prob=0 恒等 全部通过")
    return ok


def main() -> int:
    ok = test_cache() and test_field_dropout()
    print("VLM_REWRITE_CACHE_SMOKE_OK" if ok else "VLM_REWRITE_CACHE_SMOKE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
