#!/usr/bin/env python3
"""数值验证：und token VLM padding mask 修复（评审 #1）。

验证修复的核心不变量：**batch 内短样本真实 token 的输出，应与该样本
单独（无 padding）前向一致**——即 padding 位不再作为 K/V 污染注意力。

覆盖 4 条路径：
1. UniBranch.self_attn_proc 的 und<->und（K 侧真实长度）；
2. UniBranch.cross_attn_proc 的 A->B / B->A（A->B 的 K 侧真实长度）；
3. WanCrossAttention.forward(context_lens)（video/action 分支
   encode/decode 的 cross-attn 路径）；
4. UndBranch.encode 返回的 seq_lens 是逐样本真实长度（micro backbones）。

每项同时跑"旧行为（K/V 侧全长）"对照，证明测试对 padding 污染敏感。

用法：python scripts/test_und_padding_mask.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/Thinker-WM/libs" / "motus-monitor"))  # motus_monitor 包

import torch  # noqa: E402

from thinker_wm.wan.modules.model import WanAttentionBlock, WanCrossAttention, rope_params  # noqa: E402
from thinker_wm.models.uni_branch import UniBranch  # noqa: E402

DEVICE = "cuda"
DTYPE = torch.float32
ATOL = 2e-2  # flash-attn 按 bf16 计算，允许数值误差


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _build_block(dim=256, ffn_dim=1024, num_heads=8):
    torch.manual_seed(0)
    return WanAttentionBlock(
        dim=dim, ffn_dim=ffn_dim, num_heads=num_heads).to(DEVICE, DTYPE).eval()


def _build_freqs(num_heads=8, dim=256):
    sub_dim = dim // num_heads
    freqs = torch.cat([
        rope_params(1024, sub_dim - 4 * (sub_dim // 6)),
        rope_params(1024, 2 * (sub_dim // 6)),
        rope_params(1024, 2 * (sub_dim // 6)),
    ], dim=1)
    return freqs.to(DEVICE)


def _report(name: str, diff_fixed: float, diff_legacy: float) -> bool:
    ok = diff_fixed < ATOL and diff_legacy > ATOL
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: "
          f"修复后 vs 单独前向 max_diff={diff_fixed:.5f} (<{ATOL}), "
          f"旧行为(全长K/V) vs 单独前向 max_diff={diff_legacy:.5f} (应>{ATOL})")
    return ok


def test_self_attn(block, freqs) -> bool:
    """uni self-attn：und<->und 的 K 侧按真实长度 mask。"""
    torch.manual_seed(1)
    L_video, L_action, L1, L2 = 16, 5, 7, 13
    video = torch.randn(2, L_video, 256, device=DEVICE, dtype=DTYPE)
    action = torch.randn(2, L_action, 256, device=DEVICE, dtype=DTYPE)
    u1 = torch.randn(1, L1, 256, device=DEVICE, dtype=DTYPE)
    u2 = torch.randn(1, L2, 256, device=DEVICE, dtype=DTYPE)
    garbage = torch.randn(1, L2 - L1, 256, device=DEVICE, dtype=DTYPE) * 10
    und = torch.cat([torch.cat([u1, garbage], dim=1), u2], dim=0)

    vg = torch.tensor([[2, 4, 1]] * 2, device=DEVICE)
    ag = torch.tensor([[L_action, 1, 1]] * 2, device=DEVICE)
    ug = torch.tensor([[L2, 1, 1]] * 2, device=DEVICE)
    vg1, ag1, ug1 = vg[:1], ag[:1], torch.tensor([[L1, 1, 1]], device=DEVICE)

    def run(v, a, u, v_g, a_g, u_g, und_lens):
        return UniBranch.self_attn_proc(
            block.self_attn, v, v_g, a, a_g, u, u_g, freqs, freqs,
            torch.tensor([v.size(1)] * v.size(0), device=DEVICE),
            torch.tensor([a.size(1)] * a.size(0), device=DEVICE),
            torch.tensor(und_lens, device=DEVICE))

    # 单独前向（无 padding）
    _, _, und_solo = run(video[:1], action[:1], u1, vg1, ag1, ug1, [L1])
    # 修复后：K 侧真实长度
    _, _, und_fixed = run(video, action, und, vg, ag, ug, [L1, L2])
    # 旧行为：K 侧全长
    _, _, und_legacy = run(video, action, und, vg, ag, ug, [L2, L2])

    return _report(
        "self_attn und<->und",
        _max_diff(und_fixed[0, :L1], und_solo[0]),
        _max_diff(und_legacy[0, :L1], und_solo[0]))


def test_cross_attn(block) -> bool:
    """uni cross-attn：A->B（video/action 查 und，K 侧真实长度）与 B->A。"""
    torch.manual_seed(2)
    L_video, L_action, L1, L2 = 16, 5, 7, 13
    video = torch.randn(2, L_video, 256, device=DEVICE, dtype=DTYPE)
    action = torch.randn(2, L_action, 256, device=DEVICE, dtype=DTYPE)
    u1 = torch.randn(1, L1, 256, device=DEVICE, dtype=DTYPE)
    u2 = torch.randn(1, L2, 256, device=DEVICE, dtype=DTYPE)
    garbage = torch.randn(1, L2 - L1, 256, device=DEVICE, dtype=DTYPE) * 10
    und = torch.cat([torch.cat([u1, garbage], dim=1), u2], dim=0)

    def run(v, a, u, und_lens):
        return UniBranch.cross_attn_proc(
            block.cross_attn, v, a, u,
            torch.tensor([v.size(1)] * v.size(0), device=DEVICE),
            torch.tensor([a.size(1)] * a.size(0), device=DEVICE),
            torch.tensor(und_lens, device=DEVICE))

    v_solo, a_solo, u_solo = run(video[:1], action[:1], u1, [L1])
    v_fixed, a_fixed, u_fixed = run(video, action, und, [L1, L2])
    v_legacy, a_legacy, u_legacy = run(video, action, und, [L2, L2])

    # A->B：video/action 真实 token 不被 und padding 污染
    ok1 = _report(
        "cross_attn A->B (video 段)",
        _max_diff(v_fixed[0], v_solo[0]),
        _max_diff(v_legacy[0], v_solo[0]))
    ok2 = _report(
        "cross_attn A->B (action 段)",
        _max_diff(a_fixed[0], a_solo[0]),
        _max_diff(a_legacy[0], a_solo[0]))
    # B->A：und 是 Q 侧、K/V 是无 padding 的 video/action——padding 本来就不
    # 污染这条路径的真实 token（旧行为与修复后对真实 token 相同），只验证
    # 修复后真实 token 与单独前向一致这一不变量。
    diff = _max_diff(u_fixed[0, :L1], u_solo[0])
    ok3 = diff < ATOL
    print(f"[{'PASS' if ok3 else 'FAIL'}] cross_attn B->A (und 段): "
          f"修复后 vs 单独前向 max_diff={diff:.5f} (<{ATOL})")
    return ok1 and ok2 and ok3


def test_wan_cross_attention() -> bool:
    """video/action 分支 encode/decode 的 cross-attn 路径（context_lens）。"""
    torch.manual_seed(3)
    dim, num_heads = 256, 8
    ca = WanCrossAttention(dim=dim, num_heads=num_heads).to(DEVICE, DTYPE).eval()
    L_x, L1, L2 = 16, 7, 13
    x = torch.randn(2, L_x, dim, device=DEVICE, dtype=DTYPE)
    u1 = torch.randn(1, L1, dim, device=DEVICE, dtype=DTYPE)
    u2 = torch.randn(1, L2, dim, device=DEVICE, dtype=DTYPE)
    garbage = torch.randn(1, L2 - L1, dim, device=DEVICE, dtype=DTYPE) * 10
    ctx = torch.cat([torch.cat([u1, garbage], dim=1), u2], dim=0)

    out_solo = ca(x[:1], u1, None)
    out_fixed = ca(x, ctx, torch.tensor([L1, L2], device=DEVICE))
    out_legacy = ca(x, ctx, None)

    return _report(
        "WanCrossAttention context_lens",
        _max_diff(out_fixed[0], out_solo[0]),
        _max_diff(out_legacy[0], out_solo[0]))


def test_und_branch_lens() -> bool:
    """UndBranch.encode 返回逐样本真实长度（micro backbones，GPU）。"""
    from thinker_wm.train_debug.micro_backbones import build_micro_backbones
    from thinker_wm.models.und_branch import UndBranch
    from PIL import Image

    vlm_preprocess, vlm_model, _ = build_micro_backbones(
        device=DEVICE, dtype=torch.bfloat16)
    und_branch = UndBranch(
        vlm_preprocess, vlm_model, dim=256, ffn_dim=512, num_layers=1,
        vlm_dim=getattr(vlm_model, "vlm_dim", 2048),
    ).to(DEVICE, torch.bfloat16)

    pil = Image.new("RGB", (64, 64), color=(128, 64, 32))
    prompts = ["short", "a much longer task description with many more tokens"]
    und_tokens, grid_sizes, seq_lens = und_branch.encode(prompts, [pil, pil])

    L = und_tokens.size(1)
    ok = (
        seq_lens[0].item() < seq_lens[1].item()
        and seq_lens[1].item() == L
        and seq_lens.dtype == torch.long
    )
    print(f"[{'PASS' if ok else 'FAIL'}] UndBranch.encode seq_lens: "
          f"seq_lens={seq_lens.tolist()} padded_len={L} "
          f"(要求 seq_lens[0] < seq_lens[1] == padded_len)")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: 需要 CUDA（flash_attention 仅支持 CUDA）")
        return 0
    block = _build_block()
    freqs = _build_freqs()
    results = [
        test_self_attn(block, freqs),
        test_cross_attn(block),
        test_wan_cross_attention(),
        test_und_branch_lens(),
    ]
    print("=" * 80)
    print(f"UND_PADDING_MASK_TEST {'ALL_PASS' if all(results) else 'FAILED'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
