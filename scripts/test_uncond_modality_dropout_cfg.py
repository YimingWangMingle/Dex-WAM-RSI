#!/usr/bin/env python3
"""20260818_v2 dropout 统一验证（micro backbones + 合成小输入）：

1) condition dropout 已并入模态 dropout：condition_dropout_prob=1.0（弃用键）
   不再生效——und 收到的消息原样通过；modality_dropout_prompt=1.0 时
   training_step 的 und 条件为 uncond token 嵌入常量（tokenizer-only，
   不过语言模型；und_branch.get_uncond_vlm_feature），返回
   modality_dropped_prompt=1，loss 有限且 backward 正常。
2) prepare 级 VLM 半缓存：training_step 内 und 只走 encode_from_vlm_hidden
   （可训练段），不再调用完整 encode；encode() 与 precompute_vlm_features
   + encode_from_vlm_hidden 的拆分路径输出一致（数值 parity）。
3) 推理 CFG 一致：inference_step guidance_scale=5.0 时无条件前向的空文本
   经 encode 自动路由到同一 uncond 常量（训练/推理同表示）；输出形状正确、
   数值有限。

需要 CUDA（flash_attn）。用法：
    python scripts/test_uncond_modality_dropout_cfg.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "third_party/Thinker-WM/libs" / "motus-monitor"))  # motus_monitor 包

import numpy as np
import torch
import PIL.Image as Image

from thinker_wm.models.ubt_wm import MotusConfig, UBT_WM, UNCONDITIONAL_VLM_MESSAGE


def build_model(modality_dropout_prompt: float = 0.0,
                condition_dropout_prob: float = 0.0) -> UBT_WM:
    config = MotusConfig(
        action_state_dim=14,
        action_dim=14,
        video_action_freq_ratio=1,
        num_video_frames=8,
        video_height=64,
        video_width=64,
        video_encoder_layer_num=1,
        video_decoder_layer_num=1,
        action_encoder_layer_num=1,
        action_decoder_layer_num=1,
        und_layer_num=1,
        action_t_downsample=1,
        robot_configs={"piper": 14},
        robot_layout={"piper": {"offset": 0, "token_index": 0}},
        modality_dropout_prompt=modality_dropout_prompt,
        condition_dropout_prob=condition_dropout_prob,
        micro_train=True,
    )
    return UBT_WM(config).cuda()


def make_inputs(batch: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    first_frame = torch.rand(batch, 3, 64, 64, generator=g)
    video_frames = torch.rand(batch, 8, 3, 64, 64, generator=g)
    state = torch.randn(batch, 14, generator=g)
    actions = torch.randn(batch, 8, 14, generator=g)
    pils = [
        Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
        for _ in range(batch)
    ]
    messages = [f"pick up object {i}" for i in range(batch)]
    return first_frame, video_frames, state, actions, pils, messages


class UndCallRecorder:
    """记录 und_branch 各入口调用：precompute 收到的消息、from_hidden 收到的
    特征、完整 encode 的调用次数。"""

    def __init__(self, model: UBT_WM):
        self.precompute_msgs = []
        self.from_hidden_inputs = []
        self.encode_calls = 0
        und = model.world_model.und_branch

        orig_precompute = und.precompute_vlm_features
        orig_from_hidden = und.encode_from_vlm_hidden
        orig_encode = und.encode

        def wrapped_precompute(prompt_inputs, first_frame_pil, *args, **kwargs):
            self.precompute_msgs.append(list(prompt_inputs))
            return orig_precompute(prompt_inputs, first_frame_pil, *args, **kwargs)

        def wrapped_from_hidden(vlm_hidden, real_lens, *args, **kwargs):
            self.from_hidden_inputs.append((vlm_hidden, real_lens))
            return orig_from_hidden(vlm_hidden, real_lens, *args, **kwargs)

        def wrapped_encode(prompt_inputs, first_frame_pil, *args, **kwargs):
            self.encode_calls += 1
            return orig_encode(prompt_inputs, first_frame_pil, *args, **kwargs)

        und.precompute_vlm_features = wrapped_precompute
        und.encode_from_vlm_hidden = wrapped_from_hidden
        und.encode = wrapped_encode


def _train_step_inputs(model, seed=0):
    first_frame, video_frames, state, actions, pils, messages = make_inputs(2, seed=seed)
    return dict(
        first_frame=first_frame.cuda(),
        video_frames=video_frames.cuda(),
        state=state.cuda(),
        actions=actions.cuda(),
        vlm_input_messages=list(messages),
        first_frame_pils=pils,
        robot_names=["piper"] * 2,
    ), messages


def test_deprecated_condition_dropout_ignored() -> bool:
    """condition_dropout_prob=1.0（弃用键）不再生效：消息原样通过。"""
    ok = True
    model = build_model(condition_dropout_prob=1.0)
    model.train()
    recorder = UndCallRecorder(model)
    inputs, messages = _train_step_inputs(model)
    out = model.training_step(**inputs)
    assert recorder.precompute_msgs, "precompute_vlm_features 未被调用"
    if recorder.precompute_msgs[-1] != messages:
        print(f"FAIL: 弃用的 condition_dropout_prob 仍改动了消息: {recorder.precompute_msgs[-1]}")
        ok = False
    if out.get("modality_dropped_prompt", 0) != 0:
        print("FAIL: 未配 modality_dropout_prompt 不应 drop prompt")
        ok = False
    loss = out["total_loss"]
    if not torch.isfinite(loss):
        print(f"FAIL: loss 非有限: {loss}")
        ok = False
    loss.backward()
    print("deprecated condition_dropout_prob=1.0: 消息原样通过, "
          f"loss={loss.item():.4f} backward OK")
    del model
    torch.cuda.empty_cache()
    return ok


def test_modality_prompt_dropout_uncond_const() -> bool:
    """modality_dropout_prompt=1.0：und 条件 = uncond token 嵌入常量。"""
    ok = True
    model = build_model(modality_dropout_prompt=1.0)
    model.train()
    recorder = UndCallRecorder(model)
    inputs, _ = _train_step_inputs(model, seed=1)
    out = model.training_step(**inputs)
    if out.get("modality_dropped_prompt", 0) != 1:
        print(f"FAIL: modality_dropped_prompt={out.get('modality_dropped_prompt')} != 1")
        ok = False
    if recorder.encode_calls != 0:
        print(f"FAIL: 训练步不应调用完整 encode（实际 {recorder.encode_calls} 次）")
        ok = False
    assert recorder.from_hidden_inputs, "encode_from_vlm_hidden 未被调用"
    hidden, real_lens = recorder.from_hidden_inputs[-1]
    uncond_hidden, uncond_len = model.world_model.und_branch.get_uncond_vlm_feature()
    l_e = int(uncond_len[0])
    if int(hidden.shape[1]) != l_e or not torch.allclose(
        hidden[:, :l_e], uncond_hidden.expand(hidden.shape[0], -1, -1)
    ):
        print("FAIL: drop_prompt 步的 und 输入不是 uncond 常量")
        ok = False
    if not bool((real_lens == uncond_len[0]).all()):
        print(f"FAIL: drop_prompt 步 real_lens 应为 uncond 长度 {l_e}: {real_lens}")
        ok = False
    loss = out["total_loss"]
    if not torch.isfinite(loss):
        print(f"FAIL: loss 非有限: {loss}")
        ok = False
    loss.backward()
    print(f"modality_dropout_prompt=1.0: und=uncond 常量 (L_e={l_e}), "
          f"loss={loss.item():.4f} backward OK")
    del model
    torch.cuda.empty_cache()
    return ok


def test_encode_split_parity() -> bool:
    """encode() == encode_from_vlm_hidden(precompute_vlm_features(...))。"""
    ok = True
    model = build_model()
    model.eval()
    und = model.world_model.und_branch
    _, _, _, _, pils, messages = make_inputs(2, seed=2)
    with torch.no_grad():
        full = und.encode(list(messages), pils)
        hidden, real_lens = und.precompute_vlm_features(list(messages), pils)
        split = und.encode_from_vlm_hidden(hidden, real_lens)
    for name, a, b in zip(("token", "grid", "lens"), full, split):
        if isinstance(a, torch.Tensor) and a.dtype.is_floating_point:
            same = torch.allclose(a, b, atol=0, rtol=0)
        else:
            same = torch.equal(a, b)
        if not same:
            print(f"FAIL: 拆分路径与完整 encode 的 {name} 不一致")
            ok = False
    # 空文本路由：全空 batch 的 encode == uncond 常量过可训练段
    with torch.no_grad():
        uncond_hidden, uncond_len = und.get_uncond_vlm_feature()
        via_encode = und.encode([UNCONDITIONAL_VLM_MESSAGE] * 2, pils)
        via_const = und.encode_from_vlm_hidden(
            uncond_hidden.expand(2, -1, -1), uncond_len.repeat(2))
    if not torch.allclose(via_encode[0], via_const[0], atol=0, rtol=0):
        print("FAIL: 空文本 encode 未路由到 uncond 常量")
        ok = False
    print("encode 拆分 parity + 空文本 uncond 常量路由 通过")
    del model
    torch.cuda.empty_cache()
    return ok


def test_inference_cfg_consistency() -> bool:
    """推理 CFG：uncond 前向与训练 drop_prompt 同表示（encode 自动路由）。"""
    ok = True
    first_frame, _, state, _, pils, messages = make_inputs(2, seed=3)
    model = build_model()
    model.eval()
    und = model.world_model.und_branch
    encode_msgs = []
    orig_encode = und.encode

    def wrapped(prompt_inputs, first_frame_pil, *args, **kwargs):
        encode_msgs.append(list(prompt_inputs))
        return orig_encode(prompt_inputs, first_frame_pil, *args, **kwargs)

    und.encode = wrapped

    frames_1, actions_1 = model.inference_step(
        first_frame=first_frame.cuda(),
        state=state.cuda(),
        num_inference_steps=2,
        vlm_input_messages=list(messages),
        first_frame_pils=pils,
        robot_names=["piper"] * 2,
        guidance_scale=1.0,
    )
    if len(encode_msgs) != 1:
        print(f"FAIL: guidance_scale=1.0 应只编码一次 und，实际 {len(encode_msgs)} 次")
        ok = False

    torch.manual_seed(7)  # 与上面不同的初始噪声即可，只验证 CFG 通路本身
    frames_5, actions_5 = model.inference_step(
        first_frame=first_frame.cuda(),
        state=state.cuda(),
        num_inference_steps=2,
        vlm_input_messages=list(messages),
        first_frame_pils=pils,
        robot_names=["piper"] * 2,
        guidance_scale=5.0,
    )
    if len(encode_msgs) != 3:
        print(f"FAIL: guidance_scale=5.0 应再编码一次无条件 und，总调用 {len(encode_msgs)} 次")
        ok = False
    elif encode_msgs[-1] != [UNCONDITIONAL_VLM_MESSAGE] * 2:
        print(f"FAIL: 无条件 und 编码消息不对: {encode_msgs[-1]}")
        ok = False

    for name, tensor in (("frames_1", frames_1), ("actions_1", actions_1),
                         ("frames_5", frames_5), ("actions_5", actions_5)):
        if not torch.isfinite(tensor).all():
            print(f"FAIL: {name} 含非有限值")
            ok = False
    if frames_5.shape != frames_1.shape or actions_5.shape != actions_1.shape:
        print(f"FAIL: CFG 输出形状变化: {frames_5.shape} vs {frames_1.shape}")
        ok = False
    print(f"guidance=1.0: frames={tuple(frames_1.shape)} actions={tuple(actions_1.shape)} "
          f"und_encode_calls=1; guidance=5.0: und_encode_calls=2（uncond 走常量路由）, "
          "输出有限且形状一致")
    del model
    torch.cuda.empty_cache()
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: flash_attn requires CUDA; no GPU available")
        return 1
    ok = (
        test_deprecated_condition_dropout_ignored()
        and test_modality_prompt_dropout_uncond_const()
        and test_encode_split_parity()
        and test_inference_cfg_consistency()
    )
    print("UNCOND_MODALITY_DROPOUT_CFG_OK" if ok else "UNCOND_MODALITY_DROPOUT_CFG_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
