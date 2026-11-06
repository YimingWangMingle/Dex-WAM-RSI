#!/usr/bin/env python3
"""test_eef20_unified_remap.py — U=80 → U=20 列切片 remap 验证（20260815 轮）。

构造假 state_dict 走 ``UBT_WM.load_state_dict``（strict=True）验证：

1. 0812_v2 风格 ckpt（U=80、t_downsample=6）：action embedding/unembedding
   按 ``_EEF20_REMAP_COLUMNS`` 旧→新列映射 gather（每列/每行填可识别数值
   模式：列号 / 列号+块号），切片后形状正确、列顺序正确；
2. 0807 系风格 ckpt（U=80、t_downsample=1）：先 80→20 切片、再
   t_downsample 1→6（均值池化/平铺）链式 remap 后形状与数值正确；
3. fps_conditioner.*（逐层 heads 与全局单头）与 state_embedding.* 键被
   legacy 前缀丢弃（strict 加载不炸、模型无对应模块）；
4. U=20 新 ckpt（模型自身 state_dict）原样 strict 加载、参数不变。

用法：python scripts/test_eef20_unified_remap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/Thinker-WM/libs" / "motus-monitor"))  # motus_monitor 包

import torch  # noqa: E402

from thinker_wm.models.ubt_wm import MotusConfig, UBT_WM  # noqa: E402

EMB_KEY = "world_model.action_branch.encoder.action_embedding.weight"
UNW_KEY = "world_model.action_branch.decoder.action_unembedding.weight"
UNB_KEY = "world_model.action_branch.decoder.action_unembedding.bias"
STATE_EMB_W = "world_model.action_branch.encoder.state_embedding.weight"
STATE_EMB_B = "world_model.action_branch.encoder.state_embedding.bias"

OLD_U = 80
NEW_U = 20
T_DOWN = 6
COLS = list(UBT_WM._EEF20_REMAP_COLUMNS)
assert len(COLS) == NEW_U and UBT_WM._EEF20_REMAP_OLD_U == OLD_U


def _build_model() -> UBT_WM:
    """micro 替身骨干的 UBT_WM（U=20、t_downsample=6；不加载预训练权重）。"""
    config = MotusConfig(
        micro_train=True,
        micro_wan_config={"dim": 192, "ffn_dim": 384, "num_heads": 2, "num_layers": 3},
        action_dim=26,
        action_state_dim=26,
        num_video_frames=48,
        video_height=64,
        video_width=64,
        video_action_freq_ratio=6,
        action_t_downsample=T_DOWN,
        unified_action_dim=NEW_U,
        video_encoder_layer_num=1,
        video_decoder_layer_num=1,
        action_encoder_layer_num=1,
        action_decoder_layer_num=1,
        und_layer_num=1,
        gradient_checkpointing=False,
        timing_level=0,
    )
    assert config.action_chunk_size == 48 * 6, config.action_chunk_size
    return UBT_WM(config)


def _base_fake_sd(model: UBT_WM) -> dict:
    """以模型当前 state_dict 为底（保证 strict 加载其余键全覆盖）。"""
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_remap_0812v2_style(model: UBT_WM) -> None:
    """U=80 + td=6 ckpt：列切片 gather；fps/state_embedding 键丢弃。"""
    dim = int(model.state_dict()[EMB_KEY].shape[0])
    fake = _base_fake_sd(model)
    # embedding [dim,80,6,1,1]：列 c 全部填列号 c（bf16 精确表示 ≤256 整数）
    emb_old = torch.zeros(dim, OLD_U, T_DOWN, 1, 1, dtype=torch.bfloat16)
    for c in range(OLD_U):
        emb_old[:, c] = float(c)
    # unembedding [6·80,dim]/bias [6·80]：行 b*80+c 填 c+b（区分块与列）
    unw_old = torch.zeros(T_DOWN * OLD_U, dim, dtype=torch.bfloat16)
    unb_old = torch.zeros(T_DOWN * OLD_U, dtype=torch.bfloat16)
    for b in range(T_DOWN):
        for c in range(OLD_U):
            unw_old[b * OLD_U + c] = float(c + b)
            unb_old[b * OLD_U + c] = float(c + b)
    fake[EMB_KEY] = emb_old
    fake[UNW_KEY] = unw_old
    fake[UNB_KEY] = unb_old
    # 应被 legacy 前缀丢弃的键：fps_conditioner（全局单头 + 0812_v2 初版逐层
    # heads）与 state_embedding
    fake["world_model.fps_conditioner.input_proj.0.weight"] = torch.zeros(dim, 256)
    fake["world_model.fps_conditioner.head.2.weight"] = torch.zeros(dim, dim)
    fake["world_model.fps_conditioner.heads.0.2.weight"] = torch.zeros(dim, dim)
    fake[STATE_EMB_W] = torch.zeros(dim, OLD_U, 1, 1, 1)
    fake[STATE_EMB_B] = torch.zeros(dim)

    missing, unexpected = model.load_state_dict(fake, strict=True)
    assert not missing and not unexpected, (missing, unexpected)

    # fps/state 模块本身已不存在（键经 legacy 前缀丢弃，strict 不炸即证明）
    assert not hasattr(model.world_model, "fps_conditioner")
    assert not hasattr(model.world_model.action_branch.encoder, "state_embedding")

    emb = model.world_model.action_branch.encoder.action_embedding.weight
    unw = model.world_model.action_branch.decoder.action_unembedding.weight
    unb = model.world_model.action_branch.decoder.action_unembedding.bias
    assert emb.shape == (dim, NEW_U, T_DOWN, 1, 1), emb.shape
    assert unw.shape == (T_DOWN * NEW_U, dim), unw.shape
    assert unb.shape == (T_DOWN * NEW_U,), unb.shape
    # gather 列顺序：新列 j == 旧列 COLS[j]（逐元素列号模式）
    for j, c in enumerate(COLS):
        assert torch.all(emb[:, j].float() == float(c)), f"emb 列 {j} != 旧列 {c}"
        for b in range(T_DOWN):
            assert torch.all(unw[b * NEW_U + j].float() == float(c + b)), (
                f"unw 行 {b}*{NEW_U}+{j} != {c}+{b}")
            assert unb[b * NEW_U + j].float().item() == float(c + b)
    print("  [ok] 0812_v2 风格（U=80,td=6）：列切片形状/gather 顺序正确，"
          "fps_conditioner/state_embedding 键已丢弃")


def test_remap_0807_style(model: UBT_WM) -> None:
    """U=80 + td=1 ckpt（0807 系）：先 80→20 切片，再 t_downsample 1→6 链式。"""
    dim = int(model.state_dict()[EMB_KEY].shape[0])
    fake = _base_fake_sd(model)
    emb_old = torch.zeros(dim, OLD_U, 1, 1, 1, dtype=torch.bfloat16)
    for c in range(OLD_U):
        emb_old[:, c] = float(c)
    unw_old = torch.zeros(OLD_U, dim, dtype=torch.bfloat16)
    unb_old = torch.zeros(OLD_U, dtype=torch.bfloat16)
    for c in range(OLD_U):
        unw_old[c] = float(c)
        unb_old[c] = float(c)
    fake[EMB_KEY] = emb_old
    fake[UNW_KEY] = unw_old
    fake[UNB_KEY] = unb_old

    missing, unexpected = model.load_state_dict(fake, strict=True)
    assert not missing and not unexpected, (missing, unexpected)

    emb = model.world_model.action_branch.encoder.action_embedding.weight
    unw = model.world_model.action_branch.decoder.action_unembedding.weight
    unb = model.world_model.action_branch.decoder.action_unembedding.bias
    assert emb.shape == (dim, NEW_U, T_DOWN, 1, 1), emb.shape
    assert unw.shape == (T_DOWN * NEW_U, dim), unw.shape
    assert unb.shape == (T_DOWN * NEW_U,), unb.shape
    # 链式语义：切片后 t_downsample remap（emb 每 tap = 旧列/d 均值池化；
    # unw/unb 沿输出维平铺 d 份）。期望值与 remap 同序 bf16 运算（先 bf16
    # 列号、再除以 d），保证逐位一致。
    for j, c in enumerate(COLS):
        expected_tap = float(torch.tensor(float(c), dtype=torch.bfloat16) / T_DOWN)
        for tap in range(T_DOWN):
            assert torch.all(emb[:, j, tap].float() == expected_tap), (
                f"emb 列 {j} tap {tap} != {c}/{T_DOWN}")
        for b in range(T_DOWN):
            assert torch.all(unw[b * NEW_U + j].float() == float(c))
            assert unb[b * NEW_U + j].float().item() == float(c)
    print("  [ok] 0807 系风格（U=80,td=1）：80→20 切片 → 1→6 均值池化/平铺链式正确")


def test_remap_064_style(model: UBT_WM) -> None:
    """U=64 + td=1 ckpt（20260811 前）：先 64→80 零填充、再 80→20 切片、
    最后 t_downsample 1→6 平铺（链式全路径）。"""
    dim = int(model.state_dict()[EMB_KEY].shape[0])
    fake = _base_fake_sd(model)
    older_u = UBT_WM._EEF20_REMAP_OLDER_U
    emb_old = torch.zeros(dim, older_u, 1, 1, 1, dtype=torch.bfloat16)
    for c in range(older_u):
        emb_old[:, c] = float(c)
    unw_old = torch.zeros(older_u, dim, dtype=torch.bfloat16)
    unb_old = torch.zeros(older_u, dtype=torch.bfloat16)
    for c in range(older_u):
        unw_old[c] = float(c)
        unb_old[c] = float(c)
    fake[EMB_KEY] = emb_old
    fake[UNW_KEY] = unw_old
    fake[UNB_KEY] = unb_old

    missing, unexpected = model.load_state_dict(fake, strict=True)
    assert not missing and not unexpected, (missing, unexpected)

    emb = model.world_model.action_branch.encoder.action_embedding.weight
    unw = model.world_model.action_branch.decoder.action_unembedding.weight
    unb = model.world_model.action_branch.decoder.action_unembedding.bias
    assert emb.shape == (dim, NEW_U, T_DOWN, 1, 1), emb.shape
    assert unw.shape == (T_DOWN * NEW_U, dim), unw.shape
    assert unb.shape == (T_DOWN * NEW_U,), unb.shape
    # 64→80 零填充在列 64-79（含旧 rot6 列 64-69），切片后 rot6 列为 0、
    # pos/gripper 列（14-16/20/21-23/27 < 64）保留旧值；td 平铺后逐块一致。
    for j, c in enumerate(COLS):
        expected = float(c) if c < older_u else 0.0
        expected_tap = float(
            torch.tensor(expected, dtype=torch.bfloat16) / T_DOWN)
        for tap in range(T_DOWN):
            assert torch.all(emb[:, j, tap].float() == expected_tap), (
                f"emb 列 {j} tap {tap} != {expected}/{T_DOWN}")
        for b in range(T_DOWN):
            assert torch.all(unw[b * NEW_U + j].float() == expected), (
                f"unw 行 {b}*{NEW_U}+{j} != {expected}")
            assert unb[b * NEW_U + j].float().item() == expected
    print("  [ok] 064 系风格（U=64,td=1）：64→80 补零 → 80→20 切片 → 1→6 链式正确")


def test_new_u20_passthrough(model: UBT_WM) -> None:
    """U=20 新 ckpt 原样 strict 加载（remap 不触发、参数逐位不变）。"""
    branch = model.world_model.action_branch
    with torch.no_grad():
        branch.encoder.action_embedding.weight.normal_()
        branch.decoder.action_unembedding.weight.normal_()
        branch.decoder.action_unembedding.bias.normal_()
    ref = _base_fake_sd(model)
    missing, unexpected = model.load_state_dict(ref, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    now = model.state_dict()
    for key in (EMB_KEY, UNW_KEY, UNB_KEY):
        assert torch.equal(now[key], ref[key]), f"{key} 被 remap 改动"
    print("  [ok] U=20 新 ckpt：strict 原样加载、边界层参数逐位不变")


def main() -> int:
    print("[test_eef20_unified_remap] 构建 micro UBT_WM（U=20, t_downsample=6）...")
    model = _build_model()
    test_remap_0812v2_style(model)
    test_remap_0807_style(model)
    test_remap_064_style(model)
    test_new_u20_passthrough(model)
    print("EEF20_UNIFIED_REMAP_TEST ALL_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
