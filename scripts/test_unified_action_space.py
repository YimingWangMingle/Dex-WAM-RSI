#!/usr/bin/env python3
"""统一 action 空间（多机器人）纯 CPU 测试（计划 §3.9）。

覆盖：
  1) UBT_WM._scatter_to_unified：scatter/切片 round-trip（state/actions）、
     有效维 mask 正确（True 只落在 [offset, offset+dim)）、未注册机器人兜底
     （offset=0、dim=config.action_dim、不越界）；
  2) action loss 的 mask 归约公式：只在有效维上算 MSE，补 0 列不计入；
  3) RobotTokenEmbedding：形状 [B, 8, dim]、显式索引、未知机器人报 KeyError、
     槽位越界/复用报 ValueError、None/str 输入；
  4) get_robot_layout（train/train.py 经 importlib 加载）与
     _get_robot_layout（train_debug.micro_train）：unified_offset 缺省 0、
     robot_token_index 缺省按注册顺序、offset+dim>U 与 token_index 重复报错；
  5) 新增机器人只加 YAML 条目即可用：在真实 micro 配置上新增 robot
     （action_dim/unified_offset/robot_token_index），layout + scatter +
     embedding + 切片 round-trip 全链路可用（模型零改动）。

纯 CPU，无 GPU/flash_attn 依赖：
    flock /thinkerwm-pretrain/tmp/smoke_test.lock python scripts/test_unified_action_space.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/Thinker-WM/libs" / "motus-monitor"))  # motus_monitor 包

import torch
from omegaconf import OmegaConf

from thinker_wm.models.robot_tokens import RobotTokenEmbedding
from thinker_wm.models.ubt_wm import MotusConfig, UBT_WM

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  OK  {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL  {name} {detail}")


def build_stub_model(config: MotusConfig) -> UBT_WM:
    """不经 __init__ 构造 UBT_WM 实例（只测模型边界纯函数，纯 CPU）。

    __init__ 里按配置构建的策略字段在此置缺省（与 config 默认值语义一致）；
    需要 relative_eef_rot6 / gripper_hand_minmax 的用例自行设置
    ``model._repr_specs = UBT_WM._build_repr_specs(model)`` /
    ``model._norm_policy = build_gripper_hand_policy(...)``。
    """
    model = UBT_WM.__new__(UBT_WM)
    model.config = config
    model._norm_policy = None
    model._repr_specs = None
    return model


def test_scatter_round_trip() -> None:
    print("== _scatter_to_unified: scatter / 切片 round-trip + mask ==")
    config = MotusConfig(
        action_dim=26,
        unified_action_dim=64,
        robot_configs={"aloha-agilex": 14, "ARX-X5": 14, "agibot-a2-d": 26},
        robot_layout={
            "aloha-agilex": {"offset": 0, "token_index": 0},
            "ARX-X5": {"offset": 14, "token_index": 1},
            "agibot-a2-d": {"offset": 0, "token_index": 2},
        },
    )
    model = build_stub_model(config)

    # state: aloha (14 维) offset 0
    state_in = torch.randn(3, 14)
    state_out, _, mask = model._scatter_to_unified(state_in, None, ["aloha-agilex"] * 3)
    check("state shape [B,64]", state_out.shape == (3, 64), str(tuple(state_out.shape)))
    check("state 前填值保持", torch.equal(state_out[:, :14], state_in))
    check("state 补 0 列", not torch.any(state_out[:, 14:]))
    check("mask [0:14) True", bool(mask[0, :14].all()) and not bool(mask[0, 14:].any()))

    # actions: ARX-X5 (14 维) offset 14
    action_in = torch.randn(2, 10, 14)
    _, action_out, mask2 = model._scatter_to_unified(None, action_in, ["ARX-X5"] * 2)
    check("action shape [B,T,64]", action_out.shape == (2, 10, 64), str(tuple(action_out.shape)))
    check("action 写 [14:28)", torch.equal(action_out[:, :, 14:28], action_in))
    check("action 其余补 0",
          not torch.any(action_out[:, :, :14]) and not torch.any(action_out[:, :, 28:]))
    check("mask [14:28) True", bool(mask2[0, 14:28].all()) and not bool(mask2[0, 28:].any()))

    # 切片 round-trip（推理输出端语义）：U -> [offset:offset+dim]
    check("state 切片 round-trip", torch.equal(state_out[:, 0:14], state_in))
    check("action 切片 round-trip", torch.equal(action_out[:, :, 14:28], action_in))


def test_scatter_fallback() -> None:
    print("== _scatter_to_unified: 未注册机器人/robot_names=None 兜底 ==")
    config = MotusConfig(
        action_dim=26,
        unified_action_dim=64,
        robot_configs={},  # 无任何注册
        robot_layout=None,
    )
    model = build_stub_model(config)
    state_in = torch.randn(2, 26)  # dim == config.action_dim
    state_out, _, mask = model._scatter_to_unified(state_in, None, ["ego4d-null", "ghost"])
    check("未注册兜底 shape", state_out.shape == (2, 64))
    check("未注册兜底值保持（offset 0、dim 26）",
          torch.equal(state_out[:, :26], state_in) and not torch.any(state_out[:, 26:]))
    check("未注册 mask 全 26 维", bool(mask[0, :26].all()) and not bool(mask[0, 26:].any()))

    state_none = torch.randn(2, 26)
    state_out2, _, _ = model._scatter_to_unified(state_none, None, None)
    check("robot_names=None 兜底（实际维 26）",
          torch.equal(state_out2[:, :26], state_none) and not torch.any(state_out2[:, 26:]))

    # 实际张量维小于配置维时（旧调用），clamp 到实际维不越界
    small = torch.randn(2, 14)
    state_out3, _, mask3 = model._scatter_to_unified(small, None, None)
    check("实际维 14 < action_dim 26 不越界",
          state_out3.shape == (2, 64) and torch.equal(state_out3[:, :14], small)
          and not torch.any(mask3[:, 14:]))


def test_loss_mask_reduction() -> None:
    print("== action loss 有效维 mask 归约公式 ==")
    err = torch.randn(2, 5, 64) ** 2  # [B, T, U] mse reduction='none'
    mask = torch.zeros(2, 64, dtype=torch.bool)
    mask[:, :14] = True
    T = err.shape[1]
    loss = (err * mask.unsqueeze(1)).sum(dim=(1, 2)) / (mask.sum(dim=1) * T).clamp(min=1)
    manual = err[:, :, :14].mean(dim=(1, 2))
    check("mask 归约 == 有效维 MSE", torch.allclose(loss, manual), f"{loss.tolist()} vs {manual.tolist()}")
    check("逐样本 [B] 形状", loss.shape == (2,))


def test_robot_token_embedding() -> None:
    print("== RobotTokenEmbedding: 形状/显式索引/未知机器人/容量 ==")
    emb = RobotTokenEmbedding({"aloha-agilex": 0, "agibot-a2-d": 4}, dim=32)
    out = emb(["aloha-agilex", "agibot-a2-d"])
    check("形状 [B,8,dim]", tuple(out.shape) == (2, 8, 32), str(tuple(out.shape)))
    check("槽位 0 行 == 表 0 行", torch.equal(out[0], emb.table[0]))
    check("槽位 4 行 == 表 4 行", torch.equal(out[1], emb.table[4]))
    check("不同机器人 token 不同", not torch.equal(out[0], out[1]))
    check("str 单机器人输入", tuple(emb("aloha-agilex").shape) == (1, 8, 32))
    check("None 输入返回 None", emb(None) is None)
    check("空列表返回 None", emb([]) is None)

    try:
        emb(["unknown-robot"])
        check("未知机器人报 KeyError", False, "未抛异常")
    except KeyError:
        check("未知机器人报 KeyError", True)

    try:
        RobotTokenEmbedding({"a": 64}, max_robots=64, dim=8)
        check("槽位越界报 ValueError", False, "未抛异常")
    except ValueError:
        check("槽位越界报 ValueError", True)

    try:
        RobotTokenEmbedding({"a": 0, "b": 0}, max_robots=64, dim=8)
        check("槽位复用报 ValueError", False, "未抛异常")
    except ValueError:
        check("槽位复用报 ValueError", True)

    # 可训练（参数表在 state_dict 中，梯度可回流）
    emb2 = RobotTokenEmbedding({"a": 0}, dim=4)
    loss = emb2("a").sum()
    loss.backward()
    check("embedding 可训练", emb2.table.grad is not None and torch.isfinite(emb2.table.grad).all())


def _load_train_get_robot_layout():
    """尝试经 importlib 加载 train/train.py 的 get_robot_layout（失败返回 None）。"""
    try:
        import importlib.util

        sys.path.insert(0, str(_REPO_ROOT))
        sys.path.insert(0, str(_REPO_ROOT / "third_party/Thinker-WM/train"))
        spec = importlib.util.spec_from_file_location(
            "train_train_mod", str(_REPO_ROOT / "third_party/Thinker-WM/train" / "thinker_wm.train.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.get_robot_layout
    except Exception as exc:  # pragma: no cover
        print(f"  (note: 无法导入 train/train.py get_robot_layout: {type(exc).__name__}: {exc})")
        return None


def test_layout_functions() -> None:
    print("== get_robot_layout / _get_robot_layout: 缺省与校验 ==")
    from thinker_wm.train_debug.micro_train import _get_robot_layout

    # 真实 micro 配置（已加 unified_dim_map/robot_token_index；20260811 起
    # U=80 + rot groups）
    cfg = OmegaConf.load(str(_REPO_ROOT / "third_party/Thinker-WM/configs" / "micro_train_debug.yaml"))
    check("micro yaml common.unified_action_dim==80", int(cfg.common.unified_action_dim) == 80)
    layout = _get_robot_layout(cfg)
    check("layout 含全部 9 机器人", len(layout) == 9, str(len(layout)))
    check("aloha-agilex dim_map 前填 joint band",
          layout["aloha-agilex"] == {"offset": 0, "token_index": 0,
                                     "dim_map": list(range(14)),
                                     "rot_groups": None,
                                     "eef_relative": True})
    check("agibot rotvec 维由 rot group 接管",
          layout["agibot-a2-d"]["dim_map"] == [
              14, 15, 16, -1, -1, -1, 20, -1, -1, -1, -1, -1, -1,
              21, 22, 23, -1, -1, -1, 27, -1, -1, -1, -1, -1, -1]
          and layout["agibot-a2-d"]["rot_groups"] == [
              {"local": [3, 4, 5], "repr": "rotvec",
               "cols": [64, 65, 66, 67, 68, 69], "quat_order": "xyzw"},
              {"local": [16, 17, 18], "repr": "rotvec",
               "cols": [70, 71, 72, 73, 74, 75], "quat_order": "xyzw"},
          ])
    check("tienkung token_index==8 + quat rot group",
          layout["robotmind-tienkung"] == {
              "offset": 0, "token_index": 8,
              "dim_map": [14, 15, 16, -1, -1, -1, -1, 21, 22, 23,
                          -1, -1, -1, -1, 20, 27],
              "rot_groups": [
                  {"local": [3, 4, 5, 6], "repr": "quat",
                   "cols": [64, 65, 66, 67, 68, 69], "quat_order": "xyzw"},
                  {"local": [10, 11, 12, 13], "repr": "quat",
                   "cols": [70, 71, 72, 73, 74, 75], "quat_order": "xyzw"},
              ],
              "eef_relative": True})

    # 缺省分配：无 unified_dim_map / unified_offset / robot_token_index 的配置
    cfg_plain = OmegaConf.create({
        "common": {"action_dim": 26, "unified_action_dim": 64},
        "dataset": {"robot_names": {
            "a": {"action_dim": 14},
            "b": {"action_dim": 26},
        }},
    })
    layout_plain = _get_robot_layout(cfg_plain)
    check("缺省 offset=0", layout_plain["a"]["offset"] == 0 and layout_plain["b"]["offset"] == 0)
    check("缺省 dim_map=None", layout_plain["a"]["dim_map"] is None
          and layout_plain["b"]["dim_map"] is None)
    check("缺省 token_index 按注册顺序", layout_plain["a"]["token_index"] == 0
          and layout_plain["b"]["token_index"] == 1)

    # 越界与重复校验
    cfg_oob = OmegaConf.create({
        "common": {"action_dim": 26, "unified_action_dim": 64},
        "dataset": {"robot_names": {"a": {"action_dim": 14, "unified_offset": 60}}},
    })
    try:
        _get_robot_layout(cfg_oob)
        check("offset+dim>U 报 ValueError", False, "未抛异常")
    except ValueError:
        check("offset+dim>U 报 ValueError", True)

    cfg_dup = OmegaConf.create({
        "common": {"action_dim": 26, "unified_action_dim": 64},
        "dataset": {"robot_names": {
            "a": {"action_dim": 14, "robot_token_index": 0},
            "b": {"action_dim": 26, "robot_token_index": 0},
        }},
    })
    try:
        _get_robot_layout(cfg_dup)
        check("token_index 复用报 ValueError", False, "未抛异常")
    except ValueError:
        check("token_index 复用报 ValueError", True)

    # train/train.py 的同名函数（生产路径）与 micro 版本一致
    train_get_robot_layout = _load_train_get_robot_layout()
    if train_get_robot_layout is not None:
        check("thinker_wm.train.get_robot_layout 结果一致",
              train_get_robot_layout(cfg) == layout)
    else:
        print("  (skip: train/train.py get_robot_layout 未导入)")


def test_new_robot_yaml_only() -> None:
    print("== 新增机器人只加 YAML 条目即可用（回归断言） ==")
    from thinker_wm.train_debug.micro_train import _get_robot_layout

    cfg = OmegaConf.load(str(_REPO_ROOT / "third_party/Thinker-WM/configs" / "micro_train_debug.yaml"))
    new_entry = OmegaConf.create({
        "dataset": {"robot_names": {
            "my-new-robot": {
                "action_dim": 8,
                "unified_offset": 26,
                "robot_token_index": 9,
                "aliases": ["my-new-robot"],
            },
        }},
    })
    cfg_new = OmegaConf.merge(cfg, new_entry)
    layout_new = _get_robot_layout(cfg_new)
    check("新机器人进入 layout",
          layout_new.get("my-new-robot") == {"offset": 26, "token_index": 9,
                                             "dim_map": None,
                                             "rot_groups": None,
                                             "eef_relative": True})

    robot_configs = {}
    for name, value in OmegaConf.to_object(cfg_new.dataset.robot_names).items():
        robot_configs[str(name)] = int(value.get("action_dim", 26))
    config = MotusConfig(
        action_dim=26, unified_action_dim=80,
        robot_configs=robot_configs, robot_layout=layout_new,
    )
    model = build_stub_model(config)

    # scatter 新机器人 [B,8] -> [B,80] 写 [26:34]，切片 round-trip
    state_in = torch.randn(2, 8)
    state_out, _, mask = model._scatter_to_unified(state_in, None, ["my-new-robot"] * 2)
    check("新机器人 scatter shape", state_out.shape == (2, 80))
    check("新机器人写 [26:34)", torch.equal(state_out[:, 26:34], state_in))
    check("新机器人 mask [26:34)", bool(mask[0, 26:34].all()) and not bool(mask[0, 34:].any()))
    check("新机器人切片 round-trip", torch.equal(state_out[:, 26:34], state_in))

    # 新机器人 token 可用（token_index 9 -> 表第 9 行）
    emb = RobotTokenEmbedding(
        {k: v["token_index"] for k, v in layout_new.items()}, dim=16)
    out = emb(["my-new-robot"])
    check("新机器人 token 形状", tuple(out.shape) == (1, 8, 16))
    check("新机器人 token 行 == 表 9 行", torch.equal(out[0], emb.table[9]))


def test_dim_map_scatter_round_trip() -> None:
    print("== _scatter_to_unified: unified_dim_map 语义槽位 scatter / gather round-trip ==")
    from thinker_wm.models.ubt_wm import _slice_unified_actions

    # agibot 布局：本地 [L_pos3, L_rotvec3, L_grip, L_hand6, R_pos3, R_rotvec3,
    # R_grip, R_hand6] → 语义槽位（非连续、跨 band）
    agibot_map = [14, 15, 16, 17, 18, 19, 20, 28, 29, 30, 31, 32, 33,
                  21, 22, 23, 24, 25, 26, 27, 34, 35, 36, 37, 38, 39]
    config = MotusConfig(
        action_dim=26,
        unified_action_dim=64,
        robot_configs={"agibot-a2-d": 26, "aloha-agilex": 14},
        robot_layout={
            "agibot-a2-d": {"offset": 0, "token_index": 4, "dim_map": agibot_map},
            "aloha-agilex": {"offset": 0, "token_index": 0,
                             "dim_map": list(range(14))},
        },
    )
    model = build_stub_model(config)

    # 本地维 k 的值标记为 k（scatter 后列 dim_map[k] 应取值 k）
    state_in = torch.arange(26, dtype=torch.float32).unsqueeze(0)
    state_out, _, mask = model._scatter_to_unified(state_in, None, ["agibot-a2-d"])
    check("dim_map scatter shape", state_out.shape == (1, 64))
    check("dim_map scatter 逐列落位",
          all(float(state_out[0, agibot_map[k]]) == float(k) for k in range(26)))
    check("dim_map scatter 其余列补 0",
          not bool(mask[0, [c for c in range(64) if c not in agibot_map]].any()))
    check("dim_map mask 恰 26 列", int(mask[0].sum()) == 26)

    # 与 robotwin 样本混批（各自实际维齐次：agibot 26 维 / aloha 14 维分开
    # scatter，模拟两源各自的 batch）：同列共享语义一致——aloha 关节列
    # [0,14) 与 agibot EEF 列 [14,40) 互不重叠
    aloha_state = torch.arange(14, dtype=torch.float32).unsqueeze(0)
    s_out2, _, m_out2 = model._scatter_to_unified(aloha_state, None, ["aloha-agilex"])
    check("aloha dim_map 前 14 列", torch.equal(s_out2[0, :14], aloha_state[0]))
    check("aloha 14 列后补 0", not bool(m_out2[0, 14:].any()))
    check("agibot 不占关节列", not bool(mask[0, :14].any()))

    # 推理端 gather round-trip：scatter 后再 gather 应还原原始维序
    actions_in = torch.randn(2, 5, 26)
    _, a_out, _ = model._scatter_to_unified(None, actions_in, ["agibot-a2-d"] * 2)
    gathered = _slice_unified_actions(a_out, agibot_map, 0, 26)
    check("dim_map gather round-trip", torch.equal(gathered, actions_in))
    # offset 前填路径保持兼容
    sliced = _slice_unified_actions(a_out, None, 0, 26)
    check("offset 切片路径兼容", sliced.shape == (2, 5, 26))


def test_dim_map_layout_validation() -> None:
    print("== get_robot_layout: unified_dim_map 校验 ==")
    from thinker_wm.train_debug.micro_train import _get_robot_layout

    bands = {"arm": [0, 14], "eef": [14, 40], "reserve": [60, 64]}

    def make_cfg(robot_entry):
        return OmegaConf.create({
            "common": {"action_dim": 26, "unified_action_dim": 64},
            "dataset": {"robot_names": {"r": robot_entry},
                        "unified_bands": bands},
        })

    # 长度不符
    try:
        _get_robot_layout(make_cfg({"action_dim": 3, "unified_dim_map": [0, 1]}))
        check("dim_map 长度不符报 ValueError", False, "未抛异常")
    except ValueError:
        check("dim_map 长度不符报 ValueError", True)
    # 重复列
    try:
        _get_robot_layout(make_cfg({"action_dim": 3, "unified_dim_map": [0, 1, 1]}))
        check("dim_map 重复列报 ValueError", False, "未抛异常")
    except ValueError:
        check("dim_map 重复列报 ValueError", True)
    # 越界
    try:
        _get_robot_layout(make_cfg({"action_dim": 2, "unified_dim_map": [0, 64]}))
        check("dim_map 越界报 ValueError", False, "未抛异常")
    except ValueError:
        check("dim_map 越界报 ValueError", True)
    # 列不在任何已注册 band 内（如 40 ∈ band 空洞 [40,60)）
    try:
        _get_robot_layout(make_cfg({"action_dim": 2, "unified_dim_map": [0, 40]}))
        check("dim_map 列在 band 外报 ValueError", False, "未抛异常")
    except ValueError:
        check("dim_map 列在 band 外报 ValueError", True)
    # 与 unified_offset 互斥
    try:
        _get_robot_layout(make_cfg(
            {"action_dim": 2, "unified_dim_map": [0, 1], "unified_offset": 0}))
        check("dim_map 与 offset 互斥报 ValueError", False, "未抛异常")
    except ValueError:
        check("dim_map 与 offset 互斥报 ValueError", True)
    # 合法 dim_map 通过
    layout = _get_robot_layout(make_cfg({"action_dim": 3,
                                         "unified_dim_map": [14, 15, 16]}))
    check("合法 dim_map 通过", layout["r"]["dim_map"] == [14, 15, 16])


def test_rot_groups_scatter_gather_round_trip() -> None:
    print("== B1/B2 relative_eef_rot6: scatter/gather round-trip（rot group + eef pos 相对化）==")
    from thinker_wm.models import action_repr

    bands = {
        "eef_left_pos": [14, 17], "eef_left_gripper": [20, 21],
        "eef_right_pos": [21, 24], "eef_right_gripper": [27, 28],
        "eef_left_rot6": [64, 70], "eef_right_rot6": [70, 76],
    }
    agibot_map = [14, 15, 16, -1, -1, -1, 20, -1, -1, -1, -1, -1, -1,
                  21, 22, 23, -1, -1, -1, 27, -1, -1, -1, -1, -1, -1]
    rot_groups = [
        {"local": [3, 4, 5], "repr": "rotvec",
         "cols": [64, 65, 66, 67, 68, 69], "quat_order": "xyzw"},
        {"local": [16, 17, 18], "repr": "rotvec",
         "cols": [70, 71, 72, 73, 74, 75], "quat_order": "xyzw"},
    ]
    config = MotusConfig(
        action_dim=26,
        unified_action_dim=80,
        unified_bands=bands,
        action_target_repr="relative_eef_rot6",
        robot_configs={"agibot-a2-d": 26},
        robot_layout={
            "agibot-a2-d": {"offset": 0, "token_index": 4,
                            "dim_map": agibot_map, "rot_groups": rot_groups},
        },
    )
    model = build_stub_model(config)
    model._repr_specs = UBT_WM._build_repr_specs(model)

    state_in = torch.randn(2, 26) * 0.3
    actions_in = torch.randn(2, 5, 26) * 0.3
    s_out, a_out, mask = model._scatter_to_unified(
        state_in, actions_in, ["agibot-a2-d"] * 2)
    check("rot6 scatter shape [B,80]/[B,T,80]",
          s_out.shape == (2, 80) and a_out.shape == (2, 5, 80),
          f"{tuple(s_out.shape)} {tuple(a_out.shape)}")
    check("eef pos 相对化（actions - state）", torch.allclose(
        a_out[:, :, 14:17], actions_in[:, :, 0:3] - state_in[:, 0:3].unsqueeze(1), atol=1e-6))
    check("state pos 保持绝对", torch.allclose(s_out[:, 14:17], state_in[:, 0:3], atol=1e-6))
    check("夹爪列直通（不相对化）", torch.allclose(a_out[:, :, 20], actions_in[:, :, 6], atol=1e-6))
    check("mask 含 rot6 列", bool(mask[0, 64:76].all()))
    check("mask 总数 = 8 标量 + 12 rot6", int(mask[0].sum()) == 20, str(int(mask[0].sum())))
    check("输入张量未被就地修改", not torch.allclose(
        actions_in[:, :, 0:3], a_out[:, :, 14:17], atol=1e-6))

    R_s = action_repr.rotvec_to_matrix(state_in[:, 3:6])
    check("state rot 写绝对 6D", torch.allclose(
        s_out[:, 64:70], action_repr.matrix_to_6d(R_s), atol=1e-5))
    R_i = action_repr.rotvec_to_matrix(actions_in[:, :, 3:6])
    R_rel = R_s.transpose(-1, -2).unsqueeze(1) @ R_i
    check("actions rot 写相对 6D（R_sᵀR_i）", torch.allclose(
        a_out[:, :, 64:70], action_repr.matrix_to_6d(R_rel), atol=1e-5))

    # gather 还原：relative+6D → 绝对本地表示（round-trip 回原始 actions）
    gathered = model._gather_native_actions(
        a_out, state_in, ["agibot-a2-d"] * 2, agibot_map, 0, 26)
    check("gather pos 还原", torch.allclose(
        gathered[:, :, 0:3], actions_in[:, :, 0:3], atol=1e-5))
    check("gather 左臂 rot 还原", torch.allclose(
        gathered[:, :, 3:6], actions_in[:, :, 3:6], atol=1e-4),
        f"max err {(gathered[:, :, 3:6] - actions_in[:, :, 3:6]).abs().max().item()}")
    check("gather 右臂 rot 还原", torch.allclose(
        gathered[:, :, 16:19], actions_in[:, :, 16:19], atol=1e-4))
    check("gather dex 摘除维为 0", not bool(gathered[:, :, 7:12].any()))
    check("rot group 本地维集合",
          model._repr_rot_local_dims("agibot-a2-d") == {3, 4, 5, 16, 17, 18})

    # absolute_native 下同一 layout 走旧逻辑（无相对化、无 rot6 写入）
    model_abs = build_stub_model(MotusConfig(
        action_dim=26, unified_action_dim=80, unified_bands=bands,
        robot_configs={"agibot-a2-d": 26},
        robot_layout={"agibot-a2-d": {"offset": 0, "token_index": 4,
                                      "dim_map": agibot_map, "rot_groups": rot_groups}},
    ))
    _, a_abs, m_abs = model_abs._scatter_to_unified(None, actions_in, ["agibot-a2-d"] * 2)
    check("absolute_native 无 rot6 写入", not bool(m_abs[0, 64:76].any()))
    check("absolute_native pos 不相对化", torch.allclose(
        a_abs[:, :, 14:17], actions_in[:, :, 0:3], atol=1e-6))


def test_eef_relative_false_delta() -> None:
    print("== eef_relative=false（libero delta EEF）：pos 不相对化、delta 旋转直接 6D、round-trip ==")
    from thinker_wm.models import action_repr

    bands = {
        "eef_left_pos": [14, 17], "eef_left_gripper": [20, 21],
        "eef_left_rot6": [64, 70],
    }
    # libero-panda 7D：[delta pos xyz, delta rotvec, gripper]
    libero_map = [14, 15, 16, -1, -1, -1, 20]
    rot_groups = [
        {"local": [3, 4, 5], "repr": "rotvec",
         "cols": [64, 65, 66, 67, 68, 69], "quat_order": "xyzw"},
    ]
    layout_entry = {"offset": 0, "token_index": 5, "dim_map": libero_map,
                    "rot_groups": rot_groups, "eef_relative": False}
    config = MotusConfig(
        action_dim=7,
        unified_action_dim=80,
        unified_bands=bands,
        action_target_repr="relative_eef_rot6",
        robot_configs={"libero-panda": 7},
        robot_layout={"libero-panda": layout_entry},
    )
    model = build_stub_model(config)
    model._repr_specs = UBT_WM._build_repr_specs(model)
    check("eef_relative=false 时 pos_dims 为空",
          int(model._repr_specs["libero-panda"]["pos_dims"].numel()) == 0)
    check("group relative 标记为 False",
          model._repr_specs["libero-panda"]["groups"][0]["relative"] is False)

    state_in = torch.randn(2, 7) * 0.3
    actions_in = torch.randn(2, 5, 7) * 0.3
    s_out, a_out, mask = model._scatter_to_unified(
        state_in, actions_in, ["libero-panda"] * 2)
    check("delta pos 不减 state", torch.allclose(
        a_out[:, :, 14:17], actions_in[:, :, 0:3], atol=1e-6))
    check("state pos 原样写入", torch.allclose(s_out[:, 14:17], state_in[:, 0:3], atol=1e-6))
    check("gripper 直通", torch.allclose(a_out[:, :, 20], actions_in[:, :, 6], atol=1e-6))
    check("mask 含 rot6 列", bool(mask[0, 64:70].all()))
    R_s = action_repr.rotvec_to_matrix(state_in[:, 3:6])
    check("state rot 写绝对 6D", torch.allclose(
        s_out[:, 64:70], action_repr.matrix_to_6d(R_s), atol=1e-5))
    R_d = action_repr.rotvec_to_matrix(actions_in[:, :, 3:6])
    check("actions delta rot 直接 6D（无 R_sᵀ 复合）", torch.allclose(
        a_out[:, :, 64:70], action_repr.matrix_to_6d(R_d), atol=1e-5))

    gathered = model._gather_native_actions(
        a_out, state_in, ["libero-panda"] * 2, libero_map, 0, 7)
    check("gather pos round-trip", torch.allclose(
        gathered[:, :, 0:3], actions_in[:, :, 0:3], atol=1e-6))
    check("gather delta rot round-trip", torch.allclose(
        gathered[:, :, 3:6], actions_in[:, :, 3:6], atol=1e-4),
        f"max err {(gathered[:, :, 3:6] - actions_in[:, :, 3:6]).abs().max().item()}")

    # 真实 libero state 为 8D（多一个夹爪关节维）：dim_map 仅 7 项，第 8 维
    # 自动摘除，散射结果与 7D state 完全一致（native multimodal batch 接法）
    state_in8 = torch.cat([state_in, torch.randn(2, 1)], dim=-1)
    s_out8, a_out8, mask8 = model._scatter_to_unified(
        state_in8, actions_in, ["libero-panda"] * 2)
    check("8D state 散射与 7D 一致（第 8 维摘除）",
          torch.allclose(s_out8, s_out, atol=1e-6)
          and torch.allclose(a_out8, a_out, atol=1e-6)
          and bool((mask8 == mask).all()))


def test_rot_groups_layout_validation() -> None:
    print("== get_robot_layout: unified_rot_groups 校验 ==")
    from thinker_wm.train_debug.micro_train import _get_robot_layout

    bands = {
        "eef_left_pos": [14, 17], "eef_left_gripper": [20, 21],
        "eef_left_rot6": [64, 70],
    }
    base_map = [14, 15, 16, -1, -1, -1, 20]

    def make_cfg(rot_groups, dim_map=None):
        entry = {"action_dim": 7, "unified_dim_map": dim_map or base_map}
        if rot_groups is not None:
            entry["unified_rot_groups"] = rot_groups
        return OmegaConf.create({
            "common": {"action_dim": 7, "unified_action_dim": 80},
            "dataset": {"robot_names": {"r": entry}, "unified_bands": bands},
        })

    good = [{"local": [3, 4, 5], "repr": "rotvec", "cols": [64, 65, 66, 67, 68, 69]}]
    layout = _get_robot_layout(make_cfg(good))
    check("合法 rot group 通过",
          layout["r"]["rot_groups"] == [
              {"local": [3, 4, 5], "repr": "rotvec",
               "cols": [64, 65, 66, 67, 68, 69], "quat_order": "xyzw"}])

    # local 维在 dim_map 中非 -1（未被接管）
    try:
        _get_robot_layout(make_cfg(good, dim_map=[14, 15, 16, 17, 18, 19, 20]))
        check("rot local 非 -1 报 ValueError", False, "未抛异常")
    except ValueError:
        check("rot local 非 -1 报 ValueError", True)
    # cols 非 6 列
    try:
        _get_robot_layout(make_cfg(
            [{"local": [3, 4, 5], "repr": "rotvec", "cols": [64, 65, 66]}]))
        check("rot cols 非 6 列报 ValueError", False, "未抛异常")
    except ValueError:
        check("rot cols 非 6 列报 ValueError", True)
    # cols 不在 *_rot6 band 内（落在 eef_left_pos）
    try:
        _get_robot_layout(make_cfg(
            [{"local": [3, 4, 5], "repr": "rotvec",
              "cols": [14, 15, 16, 64, 65, 66]}]))
        check("rot cols 不在 rot6 band 报 ValueError", False, "未抛异常")
    except ValueError:
        check("rot cols 不在 rot6 band 报 ValueError", True)
    # cols 越界
    try:
        _get_robot_layout(make_cfg(
            [{"local": [3, 4, 5], "repr": "rotvec",
              "cols": [64, 65, 66, 67, 68, 80]}]))
        check("rot cols 越界报 ValueError", False, "未抛异常")
    except ValueError:
        check("rot cols 越界报 ValueError", True)


def main() -> int:
    torch.manual_seed(0)
    test_scatter_round_trip()
    test_scatter_fallback()
    test_loss_mask_reduction()
    test_robot_token_embedding()
    test_layout_functions()
    test_new_robot_yaml_only()
    test_dim_map_scatter_round_trip()
    test_dim_map_layout_validation()
    test_rot_groups_scatter_gather_round_trip()
    test_eef_relative_false_delta()
    test_rot_groups_layout_validation()
    if FAILURES:
        print(f"\nUNIFIED_ACTION_SPACE_FAIL: {FAILURES}")
        return 1
    print("\nUNIFIED_ACTION_SPACE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
