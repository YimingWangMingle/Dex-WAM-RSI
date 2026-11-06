#!/usr/bin/env python3
"""test_vlm_field_dropout.py — 纯 CPU 测试：VLM 四字段严格格式 + 字段 dropout。

覆盖（计划 §2.1/§2.6）：
- 16 种 drop 组合（四字段全部子集）：被 drop 字段的格式模板不出现在消息中、
  保留字段的格式模板在且保持固定顺序（Scene→Task→Action→Outcome）；
- 全 drop（4 字段全 drop）抛 ValueError；非法字段名抛 ValueError；
- 占空（placeholder）指令与禁推理指令恒在；
- sample_vlm_field_dropout：prob<=0 恒空集；prob>0 时结果恒为字段名子集、
  永不覆盖全部 4 字段（保底留 1）；prob=1 恒为恰留 1 字段。

用法：
    python scripts/test_vlm_field_dropout.py
"""

from itertools import combinations
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from thinker_wm.utils.vlm_utils import (  # noqa: E402
    MOTUS_VLM_FIELD_NAMES,
    MOTUS_VLM_FIELD_PLACEHOLDER,
    _MOTUS_VLM_FIELD_FORMATS,
    build_vlm_task_message,
    sample_vlm_field_dropout,
)

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


def main() -> int:
    fields = list(MOTUS_VLM_FIELD_NAMES)
    assert len(fields) == 4, f"expected 4 fields, got {fields}"
    assert tuple(fields) == ("Scene", "Task", "Action", "Outcome")

    task_text = "Move the mug from the shelf to the table."

    # ---- 16 种 drop 组合 ----
    all_subsets: list[set] = []
    for r in range(len(fields) + 1):
        all_subsets.extend(set(c) for c in combinations(fields, r))
    check(len(all_subsets) == 16, f"expected 16 subsets, got {len(all_subsets)}")

    for dropped in all_subsets:
        if len(dropped) == len(fields):
            # 全 drop：build 侧必须抛 ValueError（保底留 1 由采样器保证）。
            try:
                build_vlm_task_message(task_text, drop_fields=dropped)
            except ValueError:
                pass
            else:
                check(False, f"full drop {sorted(dropped)} did not raise ValueError")
            continue
        msg = build_vlm_task_message(task_text, drop_fields=dropped)
        retained = [f for f in fields if f not in dropped]
        # 被 drop 字段：整段格式模板不得出现在消息中
        for f in dropped:
            check(
                _MOTUS_VLM_FIELD_FORMATS[f] not in msg,
                f"dropped field {f} format still present",
            )
        # 保留字段：格式模板在，且按固定顺序出现
        positions = [msg.index(_MOTUS_VLM_FIELD_FORMATS[f]) for f in retained]
        check(
            positions == sorted(positions),
            f"retained field order broken for {sorted(dropped)}: {retained}",
        )
        # 保留字段标签也按固定顺序出现
        label_positions = [msg.index(f"{label}:") for label in retained]
        check(
            label_positions == sorted(label_positions),
            f"retained label order broken for {sorted(dropped)}: {retained}",
        )
        # 占空指令在
        check(MOTUS_VLM_FIELD_PLACEHOLDER in msg, "placeholder instruction missing")
        # 禁推理 / 仅输入 grounding 指令在
        check("Do NOT infer or fabricate" in msg, "no-fabrication rule missing")
        check("Ground every field ONLY in the input" in msg, "grounding rule missing")

    # ---- 非法字段名 ----
    for bad in ({"Scene", "Bogus"}, {"bogus"}):
        try:
            build_vlm_task_message(task_text, drop_fields=bad)
        except ValueError:
            pass
        else:
            check(False, f"invalid field {sorted(bad)} did not raise ValueError")

    # ---- 默认参数（只传 task_text）行为兼容：完整四字段 ----
    default_msg = build_vlm_task_message(task_text)
    for f in fields:
        check(
            _MOTUS_VLM_FIELD_FORMATS[f] in default_msg,
            f"default message missing field {f}",
        )

    # ---- sample_vlm_field_dropout ----
    check(sample_vlm_field_dropout(0.0) == frozenset(), "prob=0 not empty")
    check(sample_vlm_field_dropout(-0.5) == frozenset(), "prob<0 not empty")
    for _ in range(200):
        dropped = sample_vlm_field_dropout(0.5)
        check(set(dropped) <= set(fields), f"invalid field sampled: {dropped}")
        check(
            len(dropped) <= len(fields) - 1,
            f"sampler dropped all fields (no keep-one): {dropped}",
        )
    for _ in range(50):
        dropped = sample_vlm_field_dropout(1.0)
        check(
            len(dropped) == len(fields) - 1,
            f"prob=1 not keep-one: {dropped}",
        )

    if FAILURES:
        print(f"FAILED: {len(FAILURES)} assertion(s)")
        for msg in FAILURES:
            print(f"  - {msg}")
        return 1
    print(f"OK: 16 drop combos + sampler/validation assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
