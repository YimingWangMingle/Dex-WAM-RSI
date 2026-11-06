#!/usr/bin/env python3
"""CPU 冒烟：经 data_prepare 接入层构建 indexed mix 管线并消费若干 batch。

校验与训练完全相同的构造路径（create_training_data_pipeline）：
- batch 含 motus_model_batch_v1 契约的全部 14 个必需键
- 批次内 source_dataset 单一（训练循环的同质性要求）
- 张量形状/数值范围（像素 [0,1]、action/state 有限）

本地（micro 配置，8 源）：
    python scripts/smoke_indexed_mix_cpu.py --config configs/micro_train_debug.yaml --steps 8
H200（集群 smoke pools，10 源）：
    python scripts/smoke_indexed_mix_cpu.py --config configs/smoke_indexed_mix_cpu.yaml --steps 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from thinker_wm.data_prepare import create_training_data_pipeline  # noqa: E402
from motus_data.contracts import (  # noqa: E402
    MOTUS_MODEL_BATCH_V1_REQUIRED_KEYS,
    MOTUS_NATIVE_MULTIMODAL_BATCH_V1,
    validate_motus_model_batch_v1,
    validate_motus_native_multimodal_batch_v1,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _check_batch(batch, step: int) -> None:
    missing = [key for key in MOTUS_MODEL_BATCH_V1_REQUIRED_KEYS if key not in batch]
    assert not missing, f"step {step}: missing batch keys: {missing}"
    num_video_frames = batch["video_frames"].shape[1]
    validate_kwargs = dict(
        expected_batch_size=batch["first_frame"].shape[0],
        num_video_frames=num_video_frames,
        video_action_freq_ratio=int(batch["action_video_fps_ratios"][0]),
    )
    if batch.get("batch_adapter_profile") == MOTUS_NATIVE_MULTIMODAL_BATCH_V1:
        # state/action 原生异维源（如 libero state 8D/action 7D）走独立契约
        # （motus_model_batch_v1 的 state/action 同维断言不适用）
        validate_motus_native_multimodal_batch_v1(batch, **validate_kwargs)
    else:
        validate_motus_model_batch_v1(batch, **validate_kwargs)
    sources = batch.get("source_datasets")
    assert sources and len(set(sources)) == 1, (
        f"step {step}: mixed sources in one batch: {sources}"
    )
    for key in ("first_frame", "video_frames", "action_sequence", "initial_state"):
        tensor = batch[key]
        assert isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all(), (
            f"step {step}: {key} has non-finite values"
        )
    pixels = batch["video_frames"]
    assert 0.0 <= float(pixels.min()) and float(pixels.max()) <= 1.0, (
        f"step {step}: video_frames out of [0,1]: "
        f"[{float(pixels.min())}, {float(pixels.max())}]"
    )


def main() -> int:
    args = _parse_args()
    config = OmegaConf.load(args.config)
    pipeline = create_training_data_pipeline(
        config, rank=0, world_size=1, seed=int(args.seed)
    )
    snapshot = pipeline.snapshot()
    print(f"MIX_SOURCES {snapshot.get('mix_sources')}", flush=True)
    print(
        f"MIX_NORMALIZED_WEIGHTS {snapshot.get('mix_normalized_weights')}",
        flush=True,
    )
    consumed = 0
    with pipeline:
        for step in range(int(args.steps)):
            batch, stats = pipeline.next()
            _check_batch(batch, step)
            consumed += 1
            print(
                "BATCH_OK "
                f"step={step} source={batch['source_datasets'][0]} "
                f"mix_source={stats.get('mix_source_name')} "
                f"first_frame={tuple(batch['first_frame'].shape)} "
                f"video_frames={tuple(batch['video_frames'].shape)} "
                f"action_sequence={tuple(batch['action_sequence'].shape)} "
                f"initial_state={tuple(batch['initial_state'].shape)} "
                f"robot={batch['robot_names'][0]} "
                f"queue_size={stats.get('ready_queue_size_after_get')}",
                flush=True,
            )
    final = pipeline.snapshot()
    print(f"MIX_CONSUMED_BY_SOURCE {final.get('mix_consumed_by_source')}", flush=True)
    print(f"SMOKE_INDEXED_MIX_CPU_OK steps={consumed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
