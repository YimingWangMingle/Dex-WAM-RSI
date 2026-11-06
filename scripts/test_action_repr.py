#!/usr/bin/env python3
"""MotusBrain 对齐改造（B1/B2/B3/D1）纯 CPU 测试（20260811）。

覆盖：
  1) models/action_repr.py：rotvec/quat/rot3 ↔ 旋转矩阵 ↔ 6D round-trip
     （含大角度/近 π 分支、正交化行列式、零向量退化防护）；相对/绝对
     旋转还原数值恒等（R_s·(R_sᵀR_i) = R_i）。
  2) models/action_stats.py GripperHandNormPolicy（B3）：agibot reader-z
     逆变换还原物理量纲、夹爪维 [-1,1] min-max（q 边界）/ z-score 回退、
     未注册机器人恒等、normalize/denormalize round-trip。
  3) D1：FlowMatchScheduler shift=1 vs 6 的 σ 网格形状、
     UBT_WM._inference_sigmas 的 shift 参数与 scheduler 一致性。
  4) U 扩容 remap：_remap_unified_capacity_growth 的拷贝/置零正确性与
     非 U 维不匹配不触碰。

纯 CPU，无 GPU/flash_attn 依赖：
    python scripts/test_action_repr.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/Thinker-WM/libs" / "motus-monitor"))  # motus_monitor 包

import torch

from thinker_wm.models import action_repr
from thinker_wm.models.action_stats import build_gripper_hand_policy
from thinker_wm.models.ubt_wm import MotusConfig, UBT_WM

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  OK  {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL  {name} {detail}")


def test_rotation_conversions() -> None:
    print("== action_repr: rotvec/quat/rot3 ↔ R ↔ 6D round-trip ==")
    torch.manual_seed(0)

    rv = torch.randn(64, 3) * 0.5
    rv2 = action_repr.matrix_to_rotvec(action_repr.rotvec_to_matrix(rv))
    check("rotvec round-trip", torch.allclose(rv, rv2, atol=5e-5),
          f"max err {(rv - rv2).abs().max().item()}")

    # 大角度（含近 π）：矩阵级等价（主值 rotvec 允许不同，旋转必须相同）
    rv_big = torch.randn(64, 3)
    rv_big = rv_big / rv_big.norm(dim=-1, keepdim=True) * (torch.rand(64, 1) * 3.1 + 0.05)
    R = action_repr.rotvec_to_matrix(rv_big)
    R2 = action_repr.rotvec_to_matrix(action_repr.matrix_to_rotvec(R))
    check("大角度矩阵级等价", (R - R2).abs().max().item() < 1e-4,
          f"max err {(R - R2).abs().max().item()}")

    # 小角度数值稳定（θ→0 无 0/0）
    rv_small = torch.randn(16, 3) * 1e-8
    R_small = action_repr.rotvec_to_matrix(rv_small)
    eye = torch.eye(3).expand_as(R_small)
    check("小角度近单位阵且有限", bool(torch.isfinite(R_small).all())
          and (R_small - eye).abs().max().item() < 1e-6)

    for order in ("xyzw", "wxyz"):
        q = torch.randn(32, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        R = action_repr.quat_to_matrix(q, order)
        q2 = action_repr.matrix_to_quat(R, order)
        # 同一旋转的 q/−q 二义性：取分量误差较小者
        err = torch.minimum((q - q2).abs().max(-1).values, (q + q2).abs().max(-1).values)
        check(f"quat({order}) round-trip", float(err.max()) < 1e-4, f"max err {float(err.max())}")
        # 旋转矩阵合法（det=1、正交）
        det = torch.det(R)
        check(f"quat({order}) det≈1", (det - 1.0).abs().max().item() < 1e-4)

    # rot3 按 rotvec 处理（定义无文档假设，仅验证路径一致）
    r3 = torch.randn(8, 3) * 0.4
    check("rot3 == rotvec 路径", torch.equal(
        action_repr.repr_to_matrix(r3, "rot3"), action_repr.rotvec_to_matrix(r3)))

    # 6D：正交化 + round-trip
    R = action_repr.rotvec_to_matrix(torch.randn(32, 3))
    d6 = action_repr.matrix_to_6d(R)
    R_back = action_repr.d6_to_matrix(d6)
    check("R→6D→R round-trip", (R - R_back).abs().max().item() < 1e-5)
    noisy6 = d6 + torch.randn_like(d6) * 0.1
    R_noisy = action_repr.d6_to_matrix(noisy6)
    gram = R_noisy.transpose(-1, -2) @ R_noisy
    eye = torch.eye(3).expand_as(gram)
    check("6D 噪声输入正交化仍合法", (gram - eye).abs().max().item() < 1e-4)
    # 零向量退化防护（U 扩容新列零初始化权重的初期输出）
    R_zero = action_repr.d6_to_matrix(torch.zeros(4, 6))
    check("6D 零向量回退单位旋转", bool(torch.isfinite(R_zero).all())
          and (R_zero - torch.eye(3)).abs().max().item() < 1e-6)


def test_relative_absolute_6d() -> None:
    print("== action_repr: 相对/绝对 6D 还原恒等（B1/B2 目标变换）==")
    torch.manual_seed(1)
    for repr_name, width in (("rotvec", 3), ("quat", 4)):
        s = torch.randn(4, width)
        if repr_name == "quat":
            s = s / s.norm(dim=-1, keepdim=True)
        a = torch.randn(4, 7, width) * 0.3
        s6, a6 = action_repr.relative_rotation_6d(s, a, repr_name)
        check(f"{repr_name} state 绝对 6D 形状", s6.shape == (4, 6))
        check(f"{repr_name} action 相对 6D 形状", a6.shape == (4, 7, 6))
        # state 6D == 绝对姿态 6D
        R_s = action_repr.repr_to_matrix(s, repr_name)
        check(f"{repr_name} state 6D == matrix_to_6d(R_s)",
              (s6 - action_repr.matrix_to_6d(R_s)).abs().max().item() < 1e-5)
        back = action_repr.absolute_rotation_from_6d(s, a6, repr_name)
        R_back = action_repr.repr_to_matrix(back, repr_name)
        R_i = action_repr.repr_to_matrix(a, repr_name)
        check(f"{repr_name} 还原矩阵级恒等", (R_back - R_i).abs().max().item() < 1e-4,
              f"max err {(R_back - R_i).abs().max().item()}")
    # rotvec 主值内（小角度）逐值恒等
    s = torch.randn(4, 3) * 0.3
    a = torch.randn(4, 7, 3) * 0.3
    _, a6 = action_repr.relative_rotation_6d(s, a, "rotvec")
    back = action_repr.absolute_rotation_from_6d(s, a6, "rotvec")
    check("rotvec 逐值还原", (back - a).abs().max().item() < 1e-4)


def _agibot_layout_bands():
    bands = {
        "eef_left_pos": [14, 17], "eef_left_rotvec": [17, 20],
        "eef_left_gripper": [20, 21],
        "eef_right_pos": [21, 24], "eef_right_rotvec": [24, 27],
        "eef_right_gripper": [27, 28],
        "hand_left": [28, 34], "hand_right": [34, 40],
    }
    dim_map = [14, 15, 16, -1, -1, -1, 20, -1, -1, -1, -1, -1, -1,
               21, 22, 23, -1, -1, -1, 27, -1, -1, -1, -1, -1, -1]
    return bands, dim_map


def test_norm_policy() -> None:
    print("== GripperHandNormPolicy（B3）：reader-z 逆变换 / 夹爪回退 / round-trip ==")
    torch.manual_seed(2)
    bands, dim_map = _agibot_layout_bands()
    policy = build_gripper_hand_policy(
        {"agibot-a2-d": {"dim_map": dim_map, "rot_groups": None}}, bands)

    state = torch.randn(3, 26)
    actions = torch.randn(3, 5, 26)
    nstate, nactions = policy.normalize(state, actions, ["agibot-a2-d"] * 3)
    # agibot reader-z 逆变换：pos 维（非夹爪）= reader 值×std+mean（物理米）
    from thinker_wm.models.action_stats import ACTION_STATS
    rz_std = torch.tensor(ACTION_STATS["agibot-a2-d"]["reader_z_std"])
    rz_mean = torch.tensor(ACTION_STATS["agibot-a2-d"]["reader_z_mean"])
    check("eef pos 物理还原（reader-z 逆变换）", torch.allclose(
        nstate[:, 0:3], state[:, 0:3] * rz_std[0:3] + rz_mean[0:3], atol=1e-5))
    # 夹爪维（local 6）：q 边界缺失 → 物理 z-score 回退（reader 常量即 raw
    # 空间统计，故归一化结果恰回到 reader z 值）
    check("夹爪维 z-score 回退（≈reader z）", torch.allclose(
        nstate[:, 6], state[:, 6], atol=1e-3),
        f"max err {(nstate[:, 6] - state[:, 6]).abs().max().item()}")
    # 灵巧手摘除维恒等（-1 维：reader 恒 0 → 物理还原后为 reader 常量 mean）
    # （dex 维非 norm01 类、无 reader-z 豁免：经逆变换成为物理常量）
    check("摘除维不受 norm01 影响", torch.allclose(
        nstate[:, 7], state[:, 7] * rz_std[7] + rz_mean[7], atol=1e-5))
    # round-trip：normalize → denormalize 还原 reader 空间
    back = policy.denormalize(nactions, ["agibot-a2-d"] * 3)
    check("policy round-trip（reader 空间）", (back - actions).abs().max().item() < 1e-3,
          f"max err {(back - actions).abs().max().item()}")

    # 未注册机器人恒等
    s2, a2 = policy.normalize(state.clone(), actions.clone(), ["ghost-robot"] * 3)
    check("未注册机器人恒等", torch.equal(s2, state) and torch.equal(a2, actions))
    # robot_names=None 恒等
    s3, a3 = policy.normalize(state.clone(), actions.clone(), None)
    check("robot_names=None 恒等", torch.equal(s3, state) and torch.equal(a3, actions))


def test_norm_policy_q_bounds() -> None:
    print("== GripperHandNormPolicy（B3）：q 边界 [-1,1] min-max 映射 ==")
    bands = {"joint_left_gripper": [6, 7], "joint_right_gripper": [13, 14]}
    dim_map = list(range(14))
    # 合成 stats：夹爪维 6/13 给 q 边界（物理空间），其余维无
    import thinker_wm.models.action_stats as action_stats_mod

    key = "__test-qbot__"
    action_stats_mod.ACTION_STATS[key] = {
        "mean": [0.0] * 14,
        "std": [1.0] * 14,
        "q_low": [None] * 6 + [0.1] + [None] * 6 + [0.2],
        "q_high": [None] * 6 + [0.9] + [None] * 6 + [0.7],
        "phys_mean": [None] * 14,
        "phys_std": [None] * 14,
    }
    try:
        policy = build_gripper_hand_policy({key: {"dim_map": dim_map}}, bands)
        x = torch.zeros(2, 14)
        x[:, 6] = torch.tensor([0.1, 0.9])   # q_low / q_high
        x[:, 13] = torch.tensor([0.2, 0.7])
        nx, _ = policy.normalize(x, None, [key] * 2)
        check("q_low→-1 / q_high→+1",
              abs(float(nx[0, 6]) + 1.0) < 1e-6 and abs(float(nx[1, 6]) - 1.0) < 1e-6
              and abs(float(nx[0, 13]) + 1.0) < 1e-6 and abs(float(nx[1, 13]) - 1.0) < 1e-6,
              f"{nx[0, 6].item()} {nx[1, 6].item()}")
        check("非夹爪维恒等（物理量纲）", not bool(nx[:, :6].any())
              and not bool(nx[:, 7:13].any()))
        back = policy.denormalize(nx.clone(), [key] * 2)
        check("q 边界 round-trip", (back - x).abs().max().item() < 1e-5)
    finally:
        action_stats_mod.ACTION_STATS.pop(key, None)


def test_norm_policy_agibot_beta_parallel() -> None:
    print("== GripperHandNormPolicy（B3）：agibot-beta-parallel-gripper reader-z 逆变换 + 物理 q 边界 ==")
    torch.manual_seed(3)
    from thinker_wm.models.action_stats import ACTION_STATS
    bands = {"eef_left_gripper": [9, 10], "eef_right_gripper": [19, 20]}
    # configs/stage2_20260815_v1.yaml 的 unified_dim_map（夹爪 local 6→9、13→19）
    dim_map = [0, 1, 2, -1, -1, -1, 9,
               10, 11, 12, -1, -1, -1, 19]
    policy = build_gripper_hand_policy(
        {"agibot-beta-parallel-gripper": {"dim_map": dim_map, "rot_groups": None}},
        bands)
    stats = ACTION_STATS["agibot-beta-parallel-gripper"]
    rz_mean = torch.tensor(stats["reader_z_mean"])
    rz_std = torch.tensor(stats["reader_z_std"])

    state = torch.randn(2, 14)
    actions = torch.randn(2, 4, 14)
    nstate, _ = policy.normalize(
        state.clone(), actions.clone(), ["agibot-beta-parallel-gripper"] * 2)
    # pos 维（非夹爪）：reader z → 物理量纲（米/弧度）
    check("parallel pos z→物理（reader-z 逆变换）", torch.allclose(
        nstate[:, 0:3], state[:, 0:3] * rz_std[0:3] + rz_mean[0:3], atol=1e-5))
    # 夹爪维：z 空间旧分位边界 → 物理 mm（q*std+mean）→ min-max ±1
    x = torch.zeros(2, 14)
    x[0, 6], x[1, 6] = -0.820631, 1.491521     # local6 z 空间 q_low/q_high
    x[0, 13], x[1, 13] = -0.893770, 1.414615   # local13 z 空间 q_low/q_high
    nx, _ = policy.normalize(x, None, ["agibot-beta-parallel-gripper"] * 2)
    check("parallel 夹爪 z 边界 → ±1",
          abs(float(nx[0, 6]) + 1.0) < 1e-4 and abs(float(nx[1, 6]) - 1.0) < 1e-4
          and abs(float(nx[0, 13]) + 1.0) < 1e-4 and abs(float(nx[1, 13]) - 1.0) < 1e-4,
          f"{nx[0, 6].item()} {nx[1, 6].item()} {nx[0, 13].item()} {nx[1, 13].item()}")
    back = policy.denormalize(nx.clone(), ["agibot-beta-parallel-gripper"] * 2)
    check("parallel 夹爪 round-trip", (back - x).abs().max().item() < 1e-3,
          f"max err {(back - x).abs().max().item()}")


def test_norm_policy_agibot_beta_dexterous() -> None:
    print("== GripperHandNormPolicy（B3）：agibot-beta-dexterous-hand 仅 reader-z 逆变换（无夹爪映射）==")
    torch.manual_seed(4)
    from thinker_wm.models.action_stats import ACTION_STATS
    bands = {"eef_left_gripper": [9, 10], "eef_right_gripper": [19, 20]}
    # configs/stage2_20260815_v1.yaml 的 unified_dim_map：pos 0-2/12-14 映射，
    # rotvec 由 rot group 接管、12 手关节维 -1 摘除，gripper 列不映射
    dim_map = [0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
               10, 11, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    policy = build_gripper_hand_policy(
        {"agibot-beta-dexterous-hand": {"dim_map": dim_map, "rot_groups": None}},
        bands)
    stats = ACTION_STATS["agibot-beta-dexterous-hand"]
    rz_mean = torch.tensor(stats["reader_z_mean"])
    rz_std = torch.tensor(stats["reader_z_std"])
    state = torch.randn(2, 24)
    actions = torch.randn(2, 4, 24)
    nstate, nactions = policy.normalize(
        state.clone(), actions.clone(), ["agibot-beta-dexterous-hand"] * 2)
    # norm01 维为空 → 无任何 min-max，整向量 z→物理量纲
    check("dexterous 整向量 z→物理（reader-z 逆变换）", torch.allclose(
        nstate, state * rz_std + rz_mean, atol=1e-5))
    back = policy.denormalize(nactions.clone(), ["agibot-beta-dexterous-hand"] * 2)
    check("dexterous round-trip", (back - actions).abs().max().item() < 1e-3,
          f"max err {(back - actions).abs().max().item()}")


def _build_sched_stub(video_shift: float, action_shift: float) -> UBT_WM:
    """裸 UBT_WM 实例挂两套 scheduler（D1 测试；CPU，world_model 用 dummy 模块）。"""
    from thinker_wm.wan.utils.fm import FlowMatchScheduler

    model = UBT_WM.__new__(UBT_WM)
    torch.nn.Module.__init__(model)
    model.config = MotusConfig(flow_shift=video_shift, action_flow_shift=action_shift)
    model.dtype = torch.float32
    model.world_model = torch.nn.Linear(1, 1)  # device property 用（CPU）
    model.fm_train_scheduler = FlowMatchScheduler(
        shift=video_shift, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000)
    model.fm_train_scheduler.set_timesteps(num_inference_steps=1000, training=True)
    model.fm_train_scheduler_action = FlowMatchScheduler(
        shift=action_shift, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000)
    model.fm_train_scheduler_action.set_timesteps(num_inference_steps=1000, training=True)
    model._action_flow_shift = action_shift
    return model


def test_per_modality_timeshift() -> None:
    print("== D1：per-modality timeshift（video=6 / action=1）σ 网格 ==")
    torch.manual_seed(3)
    model = _build_sched_stub(6.0, 1.0)

    # shift=1 → σ 均匀；shift=6 → σ 偏重高噪（均值显著大于均匀）
    sig_v = model.fm_train_scheduler.sigmas
    sig_a = model.fm_train_scheduler_action.sigmas
    check("action shift=1 σ 均匀", abs(float(sig_a.mean()) - 0.5) < 0.01,
          f"mean={float(sig_a.mean())}")
    check("video shift=6 σ 偏重高噪", float(sig_v.mean()) > 0.7,
          f"mean={float(sig_v.mean())}")

    # _sample_flow_timestep：action scheduler 采样值落在 action 网格上
    _, _, sigma_a = model._sample_flow_timestep(
        2048, scheduler=model.fm_train_scheduler_action)
    flat = sigma_a.reshape(-1)
    on_grid = ((flat.unsqueeze(1) - sig_a.view(1, -1)).abs() < 1e-6).any(dim=1)
    check("action 采样落在 action 网格", bool(on_grid.all()))
    _, sigma_v, _ = model._sample_flow_timestep(2048)
    check("video 采样含高 σ 端", float(sigma_v.max()) > 0.999)

    # _inference_sigmas：shift 参数与 FlowMatchScheduler 变换一致（端点 σ=1→0）
    grid_v = model._inference_sigmas(50)
    grid_a = model._inference_sigmas(10, shift=model._action_flow_shift)
    s = torch.linspace(1.0, 0.0, 51)
    expect_v = 6.0 * s / (1.0 + 5.0 * s)
    check("video 推理网格 shift=6", (grid_v - expect_v).abs().max().item() < 1e-6)
    s10 = torch.linspace(1.0, 0.0, 11)
    check("action 推理网格 shift=1（均匀）",
          (grid_a - s10).abs().max().item() < 1e-6)
    check("网格端点 [1, 0]", abs(float(grid_v[0]) - 1.0) < 1e-6
          and float(grid_v[-1]) == 0.0)


def test_unified_capacity_remap() -> None:
    print("== U 扩容 remap：[*,64]→[*,80] 拷贝/置零，非 U 不匹配不触碰 ==")
    model = UBT_WM.__new__(UBT_WM)
    torch.nn.Module.__init__(model)
    # 20260815 U=20 轮起 _remap_unified_capacity_growth 读 config.unified_action_dim
    # 判断是否跳过 EEF20 切片键；置 80（≠20）保持本测试覆盖旧扩容路径
    model.config = MotusConfig(unified_action_dim=80)
    model.proj_in = torch.nn.Linear(80, 32)    # projector 首层：[32, 80]
    model.proj_out = torch.nn.Linear(32, 80)   # unprojector 末层：[80, 32] + bias [80]

    old_in = torch.randn(32, 64)
    old_out_w = torch.randn(64, 32)
    old_out_b = torch.randn(64)
    ckpt = {
        "proj_in.weight": old_in.clone(),
        "proj_out.weight": old_out_w.clone(),
        "proj_out.bias": old_out_b.clone(),
        "proj_in.bias": torch.randn(32),          # 形状一致：不 remap
        "other.weight": torch.randn(10, 10),      # 模型中不存在：不 remap
    }
    grown = model._remap_unified_capacity_growth(ckpt)
    check("projector 首层旧列原样拷贝", torch.equal(grown["proj_in.weight"][:, :64], old_in))
    check("projector 首层新列零初始化", not bool(grown["proj_in.weight"][:, 64:].any())
          and grown["proj_in.weight"].shape == (32, 80))
    check("unprojector 末层旧行拷贝", torch.equal(grown["proj_out.weight"][:64], old_out_w))
    check("unprojector 末层新行零初始化", not bool(grown["proj_out.weight"][64:].any())
          and grown["proj_out.weight"].shape == (80, 32))
    check("unprojector bias 扩容", torch.equal(grown["proj_out.bias"][:64], old_out_b)
          and not bool(grown["proj_out.bias"][64:].any()))
    check("形状一致键不动", torch.equal(grown["proj_in.bias"], ckpt["proj_in.bias"]))
    check("未知键不动", torch.equal(grown["other.weight"], ckpt["other.weight"]))

    # 多维不匹配不 remap（保持原样，交由调用方 skip/报错）
    model.extra = torch.nn.Linear(16, 8)
    bad = {"extra.weight": torch.randn(9, 15)}
    grown_bad = model._remap_unified_capacity_growth(bad)
    check("多维不匹配不 remap", grown_bad["extra.weight"].shape == (9, 15))


def main() -> int:
    torch.manual_seed(0)
    test_rotation_conversions()
    test_relative_absolute_6d()
    test_norm_policy()
    test_norm_policy_q_bounds()
    test_norm_policy_agibot_beta_parallel()
    test_norm_policy_agibot_beta_dexterous()
    test_per_modality_timeshift()
    test_unified_capacity_remap()
    if FAILURES:
        print(f"\nACTION_REPR_FAIL: {FAILURES}")
        return 1
    print("\nACTION_REPR_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
