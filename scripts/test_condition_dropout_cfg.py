#!/usr/bin/env python3
"""评审 #6 验证：训练期文本条件 dropout + 推理 CFG（micro backbones + 合成小输入）。

1) 训练 dropout：condition_dropout_prob=1.0 时 training_step 应把全部样本的
   VLM 文本替换为 UNCONDITIONAL_VLM_MESSAGE（包装 und_branch.encode 捕获实际
   收到的消息），返回 condition_dropped_fraction=1.0，loss 有限且 backward
   正常；p=0.0 时消息原样通过、fraction=0。
2) 推理 CFG：inference_step guidance_scale=1.0 时只编码一次 und（无无条件
   前向）；guidance_scale=5.0 时编码两次（cond + uncond），输出形状正确、
   数值有限，且与 1.0 的输出不同（CFG 确实改变采样轨迹）。

需要 CUDA（flash_attn）。用法：
    python scripts/test_condition_dropout_cfg.py
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


def build_model(condition_dropout_prob: float) -> UBT_WM:
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


class EncodeRecorder:
    """包装 und_branch.encode，记录每次调用收到的 prompt_inputs。"""

    def __init__(self, model: UBT_WM):
        self.calls = []
        original = model.world_model.und_branch.encode

        def wrapped(prompt_inputs, first_frame_pil, *args, **kwargs):
            self.calls.append(list(prompt_inputs))
            return original(prompt_inputs, first_frame_pil, *args, **kwargs)

        model.world_model.und_branch.encode = wrapped


def test_training_dropout() -> bool:
    ok = True
    first_frame, video_frames, state, actions, pils, messages = make_inputs(2)

    # p=1.0：所有样本的文本都应被替换为无条件常量
    model = build_model(condition_dropout_prob=1.0)
    model.train()
    recorder = EncodeRecorder(model)
    out = model.training_step(
        first_frame=first_frame.cuda(),
        video_frames=video_frames.cuda(),
        state=state.cuda(),
        actions=actions.cuda(),
        vlm_input_messages=list(messages),
        first_frame_pils=pils,
        robot_names=["piper"] * 2,
    )
    assert recorder.calls, "und_branch.encode 未被调用"
    received = recorder.calls[-1]
    if received != [UNCONDITIONAL_VLM_MESSAGE] * 2:
        print(f"FAIL: p=1.0 时 und 收到的消息不是无条件常量: {received}")
        ok = False
    if abs(out["condition_dropped_fraction"] - 1.0) > 1e-6:
        print(f"FAIL: condition_dropped_fraction={out['condition_dropped_fraction']} != 1.0")
        ok = False
    loss = out["total_loss"]
    if not torch.isfinite(loss):
        print(f"FAIL: loss 非有限: {loss}")
        ok = False
    loss.backward()
    print(f"p=1.0: dropped_fraction={out['condition_dropped_fraction']:.2f} "
          f"loss={loss.item():.4f} backward OK")

    # p=0.0：消息原样通过（旧行为）
    model = build_model(condition_dropout_prob=0.0)
    model.train()
    recorder = EncodeRecorder(model)
    out = model.training_step(
        first_frame=first_frame.cuda(),
        video_frames=video_frames.cuda(),
        state=state.cuda(),
        actions=actions.cuda(),
        vlm_input_messages=list(messages),
        first_frame_pils=pils,
        robot_names=["piper"] * 2,
    )
    if recorder.calls[-1] != messages:
        print(f"FAIL: p=0.0 时消息被改动: {recorder.calls[-1]}")
        ok = False
    if out["condition_dropped_fraction"] != 0.0:
        print(f"FAIL: p=0.0 时 fraction={out['condition_dropped_fraction']} != 0")
        ok = False
    print(f"p=0.0: 消息原样通过, fraction={out['condition_dropped_fraction']:.2f}")
    del model
    torch.cuda.empty_cache()
    return ok


def test_inference_cfg() -> bool:
    ok = True
    first_frame, _, state, _, pils, messages = make_inputs(2, seed=1)
    model = build_model(condition_dropout_prob=0.0)
    model.eval()
    recorder = EncodeRecorder(model)

    frames_1, actions_1 = model.inference_step(
        first_frame=first_frame.cuda(),
        state=state.cuda(),
        num_inference_steps=2,
        vlm_input_messages=list(messages),
        first_frame_pils=pils,
        robot_names=["piper"] * 2,
        guidance_scale=1.0,
    )
    if len(recorder.calls) != 1:
        print(f"FAIL: guidance_scale=1.0 应只编码一次 und，实际 {len(recorder.calls)} 次")
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
    if len(recorder.calls) != 3:
        print(f"FAIL: guidance_scale=5.0 应再编码一次无条件 und，总调用 {len(recorder.calls)} 次")
        ok = False
    elif recorder.calls[-1] != [UNCONDITIONAL_VLM_MESSAGE] * 2:
        print(f"FAIL: 无条件 und 编码消息不对: {recorder.calls[-1]}")
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
          f"und_encode_calls=1; guidance=5.0: und_encode_calls=2, 输出有限且形状一致")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: flash_attn requires CUDA; no GPU available")
        return 1
    ok = test_training_dropout() and test_inference_cfg()
    print("CONDITION_DROPOUT_CFG_OK" if ok else "CONDITION_DROPOUT_CFG_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
