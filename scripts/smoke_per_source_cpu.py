#!/usr/bin/env python3
"""逐源冒烟：对配置中的每个 source 单独构建管线并各消费若干 batch。

混合冒烟（smoke_indexed_mix_cpu.py）按权重采样，低权重源可能 32 步内
一次都消费不到；本脚本强制每个 source 都真实解码至少一个 batch，
用于新源接入后的全覆盖验证。

用法（H200）：
    OMP_NUM_THREADS=1 python scripts/smoke_per_source_cpu.py \
        --config configs/smoke_stage2_20260812_v1_cpu.yaml --batches 2
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omegaconf import OmegaConf  # noqa: E402

from thinker_wm.data_prepare import create_training_data_pipeline  # noqa: E402
from smoke_indexed_mix_cpu import _check_batch  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="只跑这些 source name（默认全部）",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = OmegaConf.load(args.config)
    names = [str(s.name) for s in config.dataset.sources]
    if args.only:
        wanted = set(args.only)
        missing = wanted - set(names)
        assert not missing, f"--only 中的 source 不在配置里: {sorted(missing)}"
        names = [n for n in names if n in wanted]

    failures: list[str] = []
    for name in names:
        cfg = OmegaConf.load(args.config)
        cfg.dataset.sources = [
            s for s in cfg.dataset.sources if str(s.name) == name
        ]
        try:
            pipeline = create_training_data_pipeline(
                cfg, rank=0, world_size=1, seed=int(args.seed)
            )
            with pipeline:
                for step in range(int(args.batches)):
                    batch, _stats = pipeline.next()
                    _check_batch(batch, step)
            print(
                "PER_SOURCE_OK "
                f"name={name} "
                f"source_dataset={batch['source_datasets'][0]} "
                f"robot={batch['robot_names'][0]} "
                f"action_dim={batch['action_sequence'].shape[-1]} "
                f"video_frames={tuple(batch['video_frames'].shape)}",
                flush=True,
            )
        except Exception:
            failures.append(name)
            print(f"PER_SOURCE_FAIL name={name}", flush=True)
            traceback.print_exc()
    if failures:
        print(f"SMOKE_PER_SOURCE_FAILED failed={failures}", flush=True)
        return 1
    print(f"SMOKE_PER_SOURCE_OK sources={len(names)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
