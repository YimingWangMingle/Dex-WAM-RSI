#!/usr/bin/env python3
"""Offline video inference/evaluation for Motus checkpoints.

This script intentionally runs single-process/single-GPU. It usually loads a
training checkpoint, samples a few dataset examples, predicts future frames
with ``model.inference_step()``, and saves GT/prediction videos for visual
review. With ``--skip-checkpoint`` it keeps only the YAML-defined pretrained
backbones plus randomly initialized Motus-specific heads/adapters.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "train"
sys.path.insert(0, str(PROJECT_ROOT / "libs" / "motus-monitor"))  # motus_monitor 包

from thinker_wm.data_prepare import create_eval_data_pipeline  # noqa: E402
from thinker_wm.utils.vlm_utils import build_vlm_rewrite_message, build_vlm_task_message  # noqa: E402
from thinker_wm.utils.vlm_utils import resolve_prompt_mode, PROMPT_MODE_TEXT_ONLY  # noqa: E402
from thinker_wm.utils.vlm_utils import resolve_vlm_prompt_style  # noqa: E402


LOG = logging.getLogger("eval_video_ckpt")


def load_config(config_path: str) -> OmegaConf:
    """Load YAML config and fill derived fields used by model/data code."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(path)
    config.common.action_chunk_size = config.common.num_video_frames * config.common.video_action_freq_ratio
    return config


def _get_robot_configs(config: OmegaConf) -> Optional[Dict[str, int]]:
    if not hasattr(config, "dataset") or not hasattr(config.dataset, "robot_names"):
        return None
    robot_names = OmegaConf.to_object(config.dataset.robot_names)
    if isinstance(robot_names, dict):
        robot_configs = {}
        for name, value in robot_names.items():
            if isinstance(value, dict):
                action_dim = value.get("action_dim", config.common.action_dim)
            else:
                action_dim = value
            robot_configs[str(name)] = int(action_dim)
        return robot_configs
    if isinstance(robot_names, list):
        return {str(name): int(config.common.action_dim) for name in robot_names}
    return None


def _get_robot_layout(config: OmegaConf) -> Optional[Dict[str, Dict[str, Any]]]:
    """Extract unified-action-space layout from dataset.robot_names
    (same logic as train/train.py get_robot_layout, incl. -1 摘除维与
    unified_rot_groups for B1/B2 relative_eef_rot6)."""
    if not hasattr(config, "dataset") or not hasattr(config.dataset, "robot_names"):
        return None
    entries = OmegaConf.to_object(config.dataset.robot_names)
    if not isinstance(entries, dict):
        return None
    U = int(getattr(config.common, "unified_action_dim", 64))
    default_dim = int(config.common.action_dim)
    bands = None
    if hasattr(config.dataset, "unified_bands"):
        raw_bands = OmegaConf.to_object(config.dataset.unified_bands)
        if isinstance(raw_bands, dict) and raw_bands:
            bands = {
                str(band): (int(span[0]), int(span[1]))
                for band, span in raw_bands.items()
            }
    used_indices = set()
    layout = {}
    for order, (name, value) in enumerate(entries.items()):
        if isinstance(value, dict):
            dim = int(value.get("action_dim", default_dim))
            offset = int(value.get("unified_offset", 0))
            token_index = int(value.get("robot_token_index", order))
            raw_map = value.get("unified_dim_map", None)
            raw_groups = value.get("unified_rot_groups", None)
        else:
            dim = default_dim
            offset = 0
            token_index = order
            raw_map = None
            raw_groups = None
        dim_map = None
        active_cols: list = []
        if raw_map is not None:
            dim_map = [int(v) for v in raw_map]
            if "unified_offset" in value:
                raise ValueError(
                    f"robot {name}: unified_dim_map 与 unified_offset 互斥，只保留 dim_map"
                )
            if len(dim_map) != dim:
                raise ValueError(
                    f"robot {name}: unified_dim_map 长度 {len(dim_map)} != action_dim {dim}"
                )
            # -1 = 摘除维（修复 #7，同 train/train.py get_robot_layout）：
            # 不参与重复/越界/band 校验
            active_cols = [c for c in dim_map if c >= 0]
            if len(set(active_cols)) != len(active_cols):
                raise ValueError(f"robot {name}: unified_dim_map 存在重复列索引")
            if any(c < -1 for c in dim_map) or (active_cols and max(active_cols) >= U):
                raise ValueError(
                    f"robot {name}: unified_dim_map 索引越界（允许 -1 摘除维或 [0, {U})）"
                )
            if bands is not None:
                for col in active_cols:
                    if not any(start <= col < end for start, end in bands.values()):
                        raise ValueError(
                            f"robot {name}: unified_dim_map 列 {col} 不在任何已注册 "
                            f"unified_bands 槽位内（先在 YAML 注册语义 band）"
                        )
        elif offset + dim > U:
            raise ValueError(
                f"robot {name}: unified_offset {offset} + action_dim {dim} "
                f"超过 common.unified_action_dim {U}"
            )
        # unified_rot_groups（B1/B2，同 train/train.py get_robot_layout）
        rot_groups = None
        if raw_groups is not None:
            if dim_map is None:
                raise ValueError(
                    f"robot {name}: unified_rot_groups 需配合 unified_dim_map 使用"
                )
            rot_groups = []
            for gi, group in enumerate(raw_groups):
                local = [int(d) for d in group["local"]]
                cols = [int(c) for c in group["cols"]]
                repr_name = str(group.get("repr", "rotvec"))
                quat_order = str(group.get("quat_order", "xyzw"))
                for d in local:
                    if d < 0 or d >= dim:
                        raise ValueError(
                            f"robot {name}: rot_groups[{gi}] local 维 {d} 超出 "
                            f"action_dim {dim}"
                        )
                    if dim_map[d] != -1:
                        raise ValueError(
                            f"robot {name}: rot_groups[{gi}] local 维 {d} 在 "
                            f"unified_dim_map 中必须为 -1（由 rot group 接管），"
                            f"got {dim_map[d]}"
                        )
                if len(cols) != 6 or len(set(cols)) != 6:
                    raise ValueError(
                        f"robot {name}: rot_groups[{gi}] cols 必须恰 6 个不重复列"
                        f"（6D 旋转），got {cols}"
                    )
                if any(c < 0 or c >= U for c in cols):
                    raise ValueError(
                        f"robot {name}: rot_groups[{gi}] cols 越界（[0, {U})），got {cols}"
                    )
                if set(cols) & set(active_cols):
                    raise ValueError(
                        f"robot {name}: rot_groups[{gi}] cols 与 unified_dim_map "
                        "有效列冲突"
                    )
                if bands is not None:
                    hit = [
                        band_name
                        for band_name, (start, end) in bands.items()
                        if all(start <= c < end for c in cols)
                        and band_name.endswith("_rot6")
                    ]
                    if len(hit) != 1:
                        raise ValueError(
                            f"robot {name}: rot_groups[{gi}] cols {cols} 未完整落在"
                            f"同一已注册 *_rot6 band 内（先在 YAML 注册语义 band）"
                        )
                rot_groups.append({
                    "local": local,
                    "repr": repr_name,
                    "cols": cols,
                    "quat_order": quat_order,
                })
        if token_index in used_indices:
            raise ValueError(
                f"robot {name}: robot_token_index {token_index} 已被其他机器人"
                "占用（禁止复用/重排）"
            )
        used_indices.add(token_index)
        layout[str(name)] = {
            "offset": offset,
            "token_index": token_index,
            "dim_map": dim_map,
            "rot_groups": rot_groups,
        }
    return layout


def create_eval_model(config: OmegaConf, load_pretrained_backbones=None) -> torch.nn.Module:
    """Create the same UBT_WM architecture as training without optimizer setup."""
    from thinker_wm.models.temporal_training import motus_temporal_kwargs
    from thinker_wm.models.ubt_wm import MotusConfig, UBT_WM

    model_config = MotusConfig(
        load_pretrained_backbones=load_pretrained_backbones,
        wan_checkpoint_path=config.model.wan.checkpoint_path,
        vae_path=config.model.wan.vae_path,
        wan_config_path=config.model.wan.config_path,
        vlm_checkpoint_path=config.model.vlm.checkpoint_path,
        video_precision=config.model.wan.precision,
        action_state_dim=config.common.state_dim,
        action_dim=config.common.action_dim,
        video_action_freq_ratio=config.common.video_action_freq_ratio,
        num_video_frames=config.common.num_video_frames,
        video_encoder_layer_num=config.model.wm.video_encoder_layer_num,
        video_decoder_layer_num=config.model.wm.video_decoder_layer_num,
        action_encoder_layer_num=config.model.wm.action_encoder_layer_num,
        action_decoder_layer_num=config.model.wm.action_decoder_layer_num,
        und_layer_num=config.model.wm.und_layer_num,
        action_t_downsample=config.model.wm.action_t_downsample,
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        video_loss_weight=config.model.loss_weights.video_loss_weight,
        action_loss_weight=config.model.loss_weights.action_loss_weight,
        freeze_action_branch=bool(getattr(config.model, "freeze_action_branch", False)),
        gradient_checkpointing=bool(getattr(config.model, "gradient_checkpointing", False)),
        checkpoint_video_branch=bool(getattr(config.model, "checkpoint_video_branch", True)),
        checkpoint_uni_branch=bool(getattr(config.model, "checkpoint_uni_branch", True)),
        checkpoint_action_branch=bool(getattr(config.model, "checkpoint_action_branch", False)),
        training_mode=getattr(config, "training_mode", "finetune"),
        robot_configs=_get_robot_configs(config),
        unified_action_dim=int(getattr(config.common, "unified_action_dim", 64)),
        robot_layout=_get_robot_layout(config),
        # 语义槽位注册表透传（B3/B1 模型边界推导用）
        unified_bands=(
            OmegaConf.to_object(config.dataset.unified_bands)
            if hasattr(config, "dataset") and hasattr(config.dataset, "unified_bands")
            else None
        ),
        # 20260811 MotusBrain 对齐（B1/B2/B3/D1）：评估模型的归一化策略、
        # 动作表示还原与推理 σ 网格必须与训练配置一致
        flow_shift=float(getattr(config.model.wm, "flow_shift", 5.0)),
        action_flow_shift=(
            float(config.model.wm.action_flow_shift)
            if getattr(config.model.wm, "action_flow_shift", None) is not None
            else None
        ),
        # 与训练同一开关。缺省 False 会让评测模型打出
        # “independent_modality_noise=False” 警告，并和 yaml 不一致。
        independent_modality_noise=bool(
            getattr(config.model.wm, "independent_modality_noise", False)
        ),
        action_norm_policy=str(getattr(config.model.wm, "action_norm_policy", "zscore")),
        action_target_repr=str(getattr(config.model.wm, "action_target_repr", "absolute_native")),
        enable_depth=bool(getattr(config.model.wm, "enable_depth", False)),
        depth_condition_mode=str(getattr(config.model.wm, "depth_condition_mode", "first_frame")),
        depth_loss_weight=float(getattr(config.model.wm, "depth_loss_weight", 1.0)),
        depth_read_by_action=bool(getattr(config.model.wm, "depth_read_by_action", False)),
        depth_read_by_video=bool(getattr(config.model.wm, "depth_read_by_video", False)),
        depth_read_by_und=bool(getattr(config.model.wm, "depth_read_by_und", False)),
        video_read_by_action=bool(getattr(config.model.wm, "video_read_by_action", True)),
        depth_detach_video_kv=bool(getattr(config.model.wm, "depth_detach_video_kv", False)),
        # 推理时 depth 初始噪声是否复用 video 的那一份。必须与训练配置一致：
        # 训练开了共享而推理另抽一份，depth 学到的"看带噪 RGB 去噪"这条路在
        # 推理失效，AbsRel 会数倍恶化而 RGB 指标毫无异常。
        depth_share_video_noise=bool(
            getattr(config.model.wm, "depth_share_video_noise", False)),
        # depth 专属 uni 克隆层数：必须与训练配置一致，否则 checkpoint 里的
        # depth_uni_blocks.* 会成为 unexpected key 被静默丢弃，depth 退回共享层。
        depth_uni_clone_layers=int(
            getattr(config.model.wm, "depth_uni_clone_layers", 0) or 0),
        vlm_text_rewrite=bool(getattr(config.model, "vlm_text_rewrite", False)),
        # 时间布局 / 历史 / 恢复条件必须与训练一致（history/recovery 参数才能加载）
        **motus_temporal_kwargs(config),
    )
    return UBT_WM(model_config)


def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_str_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None or value.strip() == "":
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _torch_load(path: Path, *, mmap: bool = False) -> Any:
    if mmap:
        # Inference may map a verified immutable checkpoint instead of eagerly
        # materializing another complete CPU copy. Older formats fail explicitly.
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_model_weights(model: torch.nn.Module, checkpoint: Path, strict: bool = True,
                        *, mmap: bool = False) -> Dict[str, Any]:
    """Load model weights from a Motus/Accelerate checkpoint directory.

    默认 strict=True（评审 #4）：评估/推理场景下缺键或多余的键必须直接
    报错，而不是带着半个随机初始化的模型继续产出"评估视频"。模型侧
    ``UBT_WM.load_state_dict`` 会先丢弃已删除模块（T5 等，评审 #10）的
    旧键，因此旧 checkpoint 在 strict 下也能加载。
    """
    candidates: List[Path] = []
    if checkpoint.is_dir():
        candidates.extend(
            [
                checkpoint / "pytorch_model_0.bin",
                checkpoint / "pytorch_model.bin",
                checkpoint / "mp_rank_00_model_states.pt",
                checkpoint / "pytorch_model" / "mp_rank_00_model_states.pt",
            ]
        )
    else:
        candidates.append(checkpoint)

    ckpt_file = next((path for path in candidates if path.exists()), None)
    if ckpt_file is None:
        tried = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(f"No supported checkpoint weight file found. Tried:\n{tried}")

    LOG.info("Loading model weights from %s", ckpt_file)
    payload = _torch_load(ckpt_file, mmap=True) if mmap else _torch_load(ckpt_file)
    state_dict = payload.get("module", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")

    # Training hooks save an unwrapped model state dict. Some older formats may
    # still include a leading module prefix.
    model_state = model.state_dict()
    if not any(key in model_state for key in state_dict.keys()):
        stripped = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                stripped[key[len("module.") :]] = value
            else:
                stripped[key] = value
        state_dict = stripped

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    info = {
        "checkpoint_file": str(ckpt_file),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "strict": bool(strict),
    }
    if missing:
        LOG.warning("Missing keys while loading checkpoint (%d): %s", len(missing), list(missing)[:20])
    if unexpected:
        LOG.warning("Unexpected keys while loading checkpoint (%d): %s", len(unexpected), list(unexpected)[:20])
    LOG.info("Checkpoint loaded: missing=%d unexpected=%d strict=%s", len(missing), len(unexpected), strict)
    return info


def _configure_eval_config(config: OmegaConf, args: argparse.Namespace) -> OmegaConf:
    """Apply small eval-only overrides without changing model/data semantics."""
    if not hasattr(config, "system"):
        config.system = {}
    if not hasattr(config, "training"):
        config.training = {}
    if not hasattr(config, "model"):
        config.model = {}
    if not hasattr(config, "logging"):
        config.logging = {}

    config.training.batch_size = 1
    config.system.num_workers = 0
    config.system.pin_memory = False
    config.system.persistent_workers = False
    config.system.prefetch_factor = None
    config.system.ready_batch_queue_enabled = False
    config.system.val_interval = 0
    config.logging.report_to = "none"
    if args.dataset_split:
        raise ValueError(
            "--dataset-split is no longer supported: train/holdout 由 split ranges 表达，"
            "请改用配置了 eval_pool 的 YAML（data_prepare.factory 会优先使用它）"
        )

    if args.num_inference_steps is not None:
        if not hasattr(config.model, "inference"):
            config.model.inference = {}
        config.model.inference.num_inference_timesteps = int(args.num_inference_steps)
    elif not hasattr(config.model, "inference") or not hasattr(config.model.inference, "num_inference_timesteps"):
        if not hasattr(config.model, "inference"):
            config.model.inference = {}
        config.model.inference.num_inference_timesteps = 20

    return config


def _select_eval_batches(pipeline: Any, args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Sequentially collect per-sample batches from the eval data pipeline.

    取代旧的按索引随机访问（indexed SamplePool 管线只支持顺序消费）。
    """
    start = max(0, int(args.start_index))
    batches: List[Dict[str, Any]] = []
    with pipeline:
        for offset in range(start + int(args.num_samples)):
            batch, _stats = pipeline.next()
            if batch is None:
                continue
            if offset >= start:
                batches.append(batch)
    if not batches:
        raise ValueError("No eval samples collected from the data pipeline")
    return batches[: int(args.num_samples)]


def _tensor_to_uint8_frames(frames: torch.Tensor) -> np.ndarray:
    """Convert [T,C,H,W] or [C,T,H,W] [0,1] tensor to uint8 [T,H,W,C]."""
    frames = frames.detach().float().cpu()
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got shape={tuple(frames.shape)}")
    if frames.shape[0] in (1, 3):
        frames = frames.permute(1, 2, 3, 0)
    elif frames.shape[1] in (1, 3):
        frames = frames.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Cannot infer channel dimension for shape={tuple(frames.shape)}")
    frames = torch.clamp(frames, 0.0, 1.0)
    return (frames.numpy() * 255.0).round().astype(np.uint8)


def _image_to_uint8(frame: torch.Tensor) -> np.ndarray:
    frame = frame.detach().float().cpu()
    if frame.ndim != 3:
        raise ValueError(f"Expected [C,H,W] image tensor, got shape={tuple(frame.shape)}")
    if frame.shape[0] in (1, 3):
        frame = frame.permute(1, 2, 0)
    frame = torch.clamp(frame, 0.0, 1.0)
    return (frame.numpy() * 255.0).round().astype(np.uint8)


def _write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, list(frames), fps=fps, macro_block_size=1)
        return
    except Exception as exc:
        LOG.warning("imageio video write failed for %s: %s; trying cv2", path, exc)

    try:
        import cv2

        height, width = frames.shape[1:3]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open {path}")
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    except Exception as exc:
        raise RuntimeError(f"Failed to write mp4 video {path}") from exc


def _label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, min(img.width, 280), 28), fill=(0, 0, 0))
    draw.text((6, 6), label, fill=(255, 255, 255))
    return np.asarray(img)


def _make_compare_frames(first: np.ndarray, gt: np.ndarray, pred: np.ndarray, include_diff: bool) -> np.ndarray:
    rows = []
    for t in range(gt.shape[0]):
        cond = first.copy()
        gt_frame = gt[t]
        pred_frame = pred[min(t, pred.shape[0] - 1)]
        panels = [
            _label_frame(cond, "condition"),
            _label_frame(gt_frame, f"gt t+{t + 1}"),
            _label_frame(pred_frame, f"pred t+{t + 1}"),
        ]
        if include_diff:
            diff = np.abs(gt_frame.astype(np.int16) - pred_frame.astype(np.int16)).astype(np.uint8)
            panels.append(_label_frame(diff, f"abs diff t+{t + 1}"))
        rows.append(np.concatenate(panels, axis=1))
    return np.stack(rows, axis=0)


def _make_compare_frames_with_conditions(
    conditions: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    include_diff: bool,
    gt_available: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    rows = []
    if gt_available is None:
        gt_available = [True] * int(gt.shape[0])
    for t in range(pred.shape[0]):
        cond = conditions[min(t, conditions.shape[0] - 1)]
        gt_frame = gt[min(t, gt.shape[0] - 1)]
        pred_frame = pred[min(t, pred.shape[0] - 1)]
        has_gt = bool(gt_available[min(t, len(gt_available) - 1)])
        panels = [
            _label_frame(cond, f"condition timeline t={t}"),
            _label_frame(gt_frame, f"gt t={t}" if has_gt else f"gt missing t={t}"),
            _label_frame(pred_frame, f"pred/condition t={t}"),
        ]
        if include_diff:
            if has_gt:
                diff = np.abs(gt_frame.astype(np.int16) - pred_frame.astype(np.int16)).astype(np.uint8)
                label = f"abs diff t={t}"
            else:
                diff = np.zeros_like(pred_frame)
                label = f"abs diff missing t={t}"
            panels.append(_label_frame(diff, label))
        rows.append(np.concatenate(panels, axis=1))
    return np.stack(rows, axis=0)


def _make_grid(first: np.ndarray, gt: np.ndarray, pred: np.ndarray, max_frames: int = 8) -> Image.Image:
    count = min(max_frames, gt.shape[0], pred.shape[0])
    h, w = gt.shape[1:3]
    canvas = Image.new("RGB", (w * count, h * 3), color=(0, 0, 0))
    for i in range(count):
        canvas.paste(Image.fromarray(first), (i * w, 0))
        canvas.paste(Image.fromarray(gt[i]), (i * w, h))
        canvas.paste(Image.fromarray(pred[i]), (i * w, h * 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), "condition", fill=(255, 255, 255))
    draw.text((5, h + 5), "ground truth", fill=(255, 255, 255))
    draw.text((5, h * 2 + 5), "prediction", fill=(255, 255, 255))
    return canvas


_LPIPS_MODEL = None


def _alexnet_checkpoint_ready() -> bool:
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    return any(hub.glob("alexnet*.pth"))


def _lpips_model():
    global _LPIPS_MODEL
    if _LPIPS_MODEL is False:
        raise RuntimeError("lpips unavailable")
    if _LPIPS_MODEL is None:
        if not _alexnet_checkpoint_ready():
            _LPIPS_MODEL = False
            raise RuntimeError("no local AlexNet weights for LPIPS")
        import lpips

        net = lpips.LPIPS(net="alex")
        net.eval()
        _LPIPS_MODEL = net
    return _LPIPS_MODEL


def _region_psnr(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float, float]:
    mse = float(((pred - gt) ** 2).mean().item())
    mae = float((pred - gt).abs().mean().item())
    psnr = float(-10.0 * math.log10(max(mse, 1e-12)))
    return mse, mae, psnr


def _gaussian_window(window_size: int, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    gauss = gauss / gauss.sum()
    kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    return kernel.expand(channels, 1, window_size, window_size).contiguous()


def _region_ssim(pred: torch.Tensor, gt: torch.Tensor) -> Optional[float]:
    """Mean SSIM over frames. Window 11, k1/k2=0.01/0.03, data range [0,1]."""
    window_size = 11
    if pred.shape[-2] < window_size or pred.shape[-1] < window_size:
        return None
    channels = int(pred.shape[1])
    window = _gaussian_window(window_size, channels)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window_size // 2
    scores = []
    for i in range(pred.shape[0]):
        x = pred[i : i + 1]
        y = gt[i : i + 1]
        mu_x = F.conv2d(x, window, padding=pad, groups=channels)
        mu_y = F.conv2d(y, window, padding=pad, groups=channels)
        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        sig_x2 = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x2
        sig_y2 = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y2
        sig_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy
        ssim_map = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2)
        )
        scores.append(float(ssim_map.mean().item()))
    return float(np.mean(scores))


def _region_lpips(pred: torch.Tensor, gt: torch.Tensor) -> Optional[float]:
    try:
        net = _lpips_model()
    except Exception:
        return None
    x = (pred * 2.0 - 1.0).clamp(-1.0, 1.0)
    y = (gt * 2.0 - 1.0).clamp(-1.0, 1.0)
    scores = []
    with torch.no_grad():
        for i in range(x.shape[0]):
            scores.append(float(net(x[i : i + 1], y[i : i + 1]).mean().item()))
    return float(np.mean(scores))


def _compute_video_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    view_slots: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
) -> Dict[str, Any]:
    pred = pred.detach().float().cpu()
    gt = gt.detach().float().cpu()
    min_t = min(pred.shape[0], gt.shape[0])
    pred = pred[:min_t]
    gt = gt[:min_t]
    mse_per_frame = ((pred - gt) ** 2).flatten(1).mean(dim=1)
    mae_per_frame = (pred - gt).abs().flatten(1).mean(dim=1)
    mse, mae, psnr = _region_psnr(pred, gt)
    result = {
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "mse_per_frame": [float(x) for x in mse_per_frame.tolist()],
        "mae_per_frame": [float(x) for x in mae_per_frame.tolist()],
        "ssim": None,
        "lpips": None,
    }
    if view_slots:
        h, w = int(pred.shape[-2]), int(pred.shape[-1])
        for name, box in view_slots.items():
            r0, r1, c0, c1 = (int(v) for v in box)
            if r1 <= r0 or c1 <= c0 or r1 > h or c1 > w:
                continue
            p = pred[:, :, r0:r1, c0:c1]
            g = gt[:, :, r0:r1, c0:c1]
            rmse, rmae, rpsnr = _region_psnr(p, g)
            result[f"mse_{name}"] = rmse
            result[f"mae_{name}"] = rmae
            result[f"psnr_{name}"] = rpsnr
    return result


@torch.no_grad()
def _run_chunked_teacher_forcing(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    num_inference_steps: int,
    chunk_size: int,
    num_chunks: int,
    guidance_scale: float = 1.0,
    action_inference_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Predict short chunks and stitch condition frames into the timeline.

    For frames [frame0, frame1, ...], chunk 0 emits
    [GT frame0, pred frame1..frame8], chunk 1 emits
    [GT frame9, pred frame10..frame17], and so on. This keeps the output
    timeline continuous in 9-frame units while periodically resetting the model
    with a real condition frame.
    """
    gt_all = batch["video_frames"][0].detach().cpu().float()
    first_frame = batch["first_frame"][0].detach().cpu().float()
    chunk_size = int(chunk_size)
    num_chunks = int(num_chunks)
    if chunk_size <= 0 or num_chunks <= 0:
        raise ValueError("chunk_size and num_chunks must be positive")

    timeline_pred: List[torch.Tensor] = []
    timeline_gt: List[torch.Tensor] = []
    timeline_conditions: List[torch.Tensor] = []
    gt_available: List[bool] = []
    metric_pred: List[torch.Tensor] = []
    metric_gt: List[torch.Tensor] = []
    chunk_records: List[Dict[str, Any]] = []
    predicted_actions_chunks: List[torch.Tensor] = []
    unit = chunk_size + 1

    def frame_at(global_t: int) -> Optional[torch.Tensor]:
        if global_t == 0:
            return first_frame
        idx = global_t - 1
        if 0 <= idx < gt_all.shape[0]:
            return gt_all[idx]
        return None

    for chunk_idx in range(num_chunks):
        condition_global_t = chunk_idx * unit
        condition_chw = frame_at(condition_global_t)
        if condition_chw is None:
            break
        condition_label = "first_frame" if condition_global_t == 0 else f"gt_frame_t{condition_global_t}"

        local_batch = dict(batch)
        local_batch["first_frame"] = condition_chw.unsqueeze(0).to(dtype=batch["first_frame"].dtype)
        predicted_frames_cthw, predicted_actions = _run_one_sample(
            model=model,
            batch=local_batch,
            device=device,
            dtype=dtype,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            action_inference_params=action_inference_params,
        )
        if predicted_frames_cthw is None:
            # 视频模态缺失（评审 #4）：该 chunk 无预测帧（inference_step 返回
            # None），teacher forcing 依赖视频条件帧，无法继续，提前结束。
            LOG.warning(
                "chunk %d: predicted_frames=None (video modality missing); "
                "stopping teacher-forcing chunk loop", chunk_idx,
            )
            break
        pred_tchw = predicted_frames_cthw[0].detach().cpu().permute(1, 0, 2, 3).float()
        pred_tchw = pred_tchw[:chunk_size]
        pred_len = int(pred_tchw.shape[0])

        # Emit the real condition frame at its true timeline position.
        timeline_pred.append(condition_chw)
        timeline_gt.append(condition_chw)
        timeline_conditions.append(condition_chw)
        gt_available.append(True)

        kept_pred = 0
        kept_gt = 0
        for offset in range(pred_len):
            global_t = condition_global_t + 1 + offset
            pred_frame = pred_tchw[offset]
            gt_frame = frame_at(global_t)
            timeline_pred.append(pred_frame)
            timeline_conditions.append(condition_chw)
            kept_pred += 1
            if gt_frame is None:
                timeline_gt.append(torch.zeros_like(pred_frame))
                gt_available.append(False)
            else:
                timeline_gt.append(gt_frame)
                gt_available.append(True)
                metric_pred.append(pred_frame)
                metric_gt.append(gt_frame)
                kept_gt += 1

        if predicted_actions is not None:
            predicted_actions_chunks.append(predicted_actions.detach().cpu())
        chunk_records.append(
            {
                "chunk_index": int(chunk_idx),
                "condition": condition_label,
                "condition_global_t": int(condition_global_t),
                "target_start_global_t": int(condition_global_t + 1),
                "target_end_global_t_inclusive": int(condition_global_t + kept_pred),
                "kept_pred_frames": int(kept_pred),
                "kept_gt_frames": int(kept_gt),
            }
        )

    if not timeline_pred:
        raise ValueError(
            f"No valid chunks for chunk_size={chunk_size}, num_chunks={num_chunks}, "
            f"available_future_frames={gt_all.shape[0]}"
        )

    pred = torch.stack(timeline_pred, dim=0)
    gt = torch.stack(timeline_gt, dim=0)
    conditions = torch.stack(timeline_conditions, dim=0)
    metric_pred_tchw = torch.stack(metric_pred, dim=0) if metric_pred else pred[:0]
    metric_gt_tchw = torch.stack(metric_gt, dim=0) if metric_gt else gt[:0]
    actions = torch.cat(predicted_actions_chunks, dim=0) if predicted_actions_chunks else None
    return {
        "pred_tchw": pred,
        "gt_tchw": gt,
        "conditions_tchw": conditions,
        "metric_pred_tchw": metric_pred_tchw,
        "metric_gt_tchw": metric_gt_tchw,
        "gt_available": gt_available,
        "predicted_actions": actions,
        "chunks": chunk_records,
    }


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=str)


def _move_vlm_inputs(vlm_inputs: Any, device: torch.device) -> Any:
    if vlm_inputs is None:
        return None
    if isinstance(vlm_inputs, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in vlm_inputs.items()}
    return vlm_inputs


@torch.no_grad()
def _run_one_sample(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    num_inference_steps: int,
    guidance_scale: float = 1.0,
    action_inference_params: Optional[Dict[str, Any]] = None,
    return_depth: bool = False,
    teacher_force_depth: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    first_frame = batch["first_frame"].to(device=device, dtype=dtype)
    state = batch.get("initial_state")
    if state is not None:
        # 与训练 / periodic_eval 一致：state 保持 fp32，z-score 统计量是 fp32。
        state = state.to(device=device, dtype=torch.float32)
    language_embeddings = batch.get("language_embedding")
    if language_embeddings is not None:
        language_embeddings = language_embeddings.to(device=device, dtype=dtype)
    vlm_inputs = _move_vlm_inputs(batch.get("vlm_inputs"), device)

    depth_first = None
    present = batch.get("depth_present")
    want_depth = bool(return_depth)
    if want_depth and batch.get("depth_first") is not None:
        if present is None or bool(present.reshape(-1)[0].item()):
            depth_first = batch["depth_first"].to(device=device, dtype=dtype)
        else:
            want_depth = False

    condition_depth_frames = None
    if (
        teacher_force_depth
        and want_depth
        and batch.get("depth_frames") is not None
    ):
        condition_depth_frames = batch["depth_frames"].to(device=device, dtype=dtype)

    out = model.inference_step(
        first_frame=first_frame,
        state=state,
        num_inference_steps=int(num_inference_steps),
        language_embeddings=language_embeddings,
        vlm_inputs=vlm_inputs,
        first_frame_pils=batch.get("first_frame_pils"),
        t5_input_messages=batch.get("t5_input_messages"),
        vlm_input_messages=batch.get("vlm_input_messages"),
        robot_names=batch.get("robot_names"),
        guidance_scale=float(guidance_scale),
        depth_first=depth_first,
        return_depth=want_depth,
        condition_depth_frames=condition_depth_frames,
        **(action_inference_params or {}),
    )
    if return_depth:
        if want_depth:
            return out
        predicted_frames, predicted_actions = out
        return predicted_frames, predicted_actions, None
    return out


def _sample_meta(batch: Dict[str, Any], sample_index: int, dataset_index: int) -> Dict[str, Any]:
    def first_value(key: str) -> Any:
        value = batch.get(key)
        if isinstance(value, list):
            return value[0] if value else None
        if torch.is_tensor(value):
            return value[0].detach().cpu().tolist() if value.ndim > 0 else value.item()
        return value

    return {
        "sample_index": int(sample_index),
        "dataset_index": int(dataset_index),
        "manifest_id": first_value("manifest_ids"),
        "manifest_task": first_value("manifest_tasks"),
        "source_dataset": first_value("source_datasets"),
        "robot_name": first_value("robot_names"),
        "t5_input_message": first_value("t5_input_messages"),
        "vlm_input_message": first_value("vlm_input_messages"),
        "first_frame_shape": list(batch["first_frame"].shape),
        "video_frames_shape": list(batch["video_frames"].shape),
        "action_sequence_shape": list(batch["action_sequence"].shape)
        if "action_sequence" in batch and batch["action_sequence"] is not None
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Motus video generation checkpoint on dataset samples.")
    parser.add_argument("--config", required=True, help="Training YAML config used to build model/data.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory or checkpoint weight file.")
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help=(
            "Do not load a Motus training checkpoint. This is a pretrained-backbone baseline: "
            "WAN/Qwen3-VL are loaded from YAML paths, while Motus-specific action/understanding "
            "heads and prompts remain randomly initialized."
        ),
    )
    parser.add_argument("--output", required=True, help="Output directory for videos and metrics.")
    parser.add_argument("--num-samples", type=int, default=8, help="Maximum number of samples to evaluate.")
    parser.add_argument("--sample-indices", default=None, help="Comma-separated global dataset indices.")
    parser.add_argument("--manifest-ids", default=None, help="Comma-separated manifest ids to locate and evaluate.")
    parser.add_argument("--manifest-scan-limit", type=int, default=None, help="Limit records scanned for manifest id lookup.")
    parser.add_argument(
        "--dataset-split",
        default=None,
        choices=["train", "val", "test"],
        help="Override all dataset child split values before creating the eval dataset.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Sequential sampling start index.")
    parser.add_argument("--random", action="store_true", help="Randomly sample dataset indices.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Inference dtype for inputs.")
    parser.add_argument("--num-inference-steps", type=int, default=20, help="Denoising steps for model.inference_step().")
    parser.add_argument(
        "--chunked-teacher-forcing",
        action="store_true",
        help="Reset condition with GT frames every chunk instead of evaluating one long free rollout.",
    )
    parser.add_argument("--chunk-size", type=int, default=8, help="Frames kept from each chunked prediction.")
    parser.add_argument("--num-chunks", type=int, default=6, help="Number of teacher-forced chunks to stitch.")
    parser.add_argument("--fps", type=int, default=6, help="Saved mp4 FPS.")
    parser.add_argument("--include-diff", action="store_true", help="Add abs-diff panel to compare.mp4.")
    parser.add_argument(
        "--no-strict-load",
        action="store_true",
        help="加载 checkpoint 时允许 missing/unexpected 键（默认 strict=True，"
             "评审 #4；旧行为可用本开关找回）。",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()
    if not args.skip_checkpoint and not args.checkpoint:
        parser.error("--checkpoint is required unless --skip-checkpoint is set")
    if args.sample_indices or args.manifest_ids or args.random:
        parser.error(
            "--sample-indices/--manifest-ids/--random 依赖旧 loader 的按索引随机访问，"
            "indexed SamplePool 管线只支持顺序消费；请用 --start-index/--num-samples"
        )

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    config = _configure_eval_config(load_config(args.config), args)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    LOG.info("Creating model")
    model = create_eval_model(config)
    if args.skip_checkpoint:
        load_info = {
            "checkpoint_file": None,
            "skipped": True,
            "note": (
                "Motus checkpoint was not loaded. WAN/Qwen3-VL backbones come from YAML "
                "pretrained paths; Motus-specific action/understanding heads and prompts are random."
            ),
        }
        LOG.warning(load_info["note"])
    else:
        load_info = _load_model_weights(model, Path(args.checkpoint), strict=not args.no_strict_load)
    model.to(device)
    model.eval()

    LOG.info("Creating eval data pipeline")
    pipeline = create_eval_data_pipeline(config)
    batches = _select_eval_batches(pipeline, args)
    LOG.info("Selected %d eval samples (start_index=%d)", len(batches), int(args.start_index))

    # CFG 引导强度（评审 #6，仅 video）：缺省 1.0 = 关闭
    guidance_scale = float(getattr(config.model.inference, "guidance_scale", 1.0))
    LOG.info("CFG guidance_scale=%.2f", guidance_scale)
    # action 侧推理参数（修复 #1/#3b/#9c）：action CFG（默认 1.0=不做）、
    # 独立去噪步数（默认 None=与 video 同）、时序平滑（默认 0=关闭）
    _inference_cfg = config.model.inference
    _action_steps = getattr(_inference_cfg, "action_num_inference_timesteps", None)
    action_inference_params = {
        "action_guidance_scale": float(getattr(_inference_cfg, "action_guidance_scale", 1.0)),
        "action_num_inference_steps": int(_action_steps) if _action_steps is not None else None,
        "action_smooth_alpha": float(getattr(_inference_cfg, "action_temporal_smooth_alpha", 0.0)),
        "depth_smooth_alpha": float(getattr(_inference_cfg, "depth_temporal_smooth_alpha", 0.0)),
    }
    LOG.info("action inference params=%s", action_inference_params)

    metrics_path = output_dir / "metrics.jsonl"
    records: List[Dict[str, Any]] = []
    start_time = time.time()
    with metrics_path.open("w", encoding="utf-8") as metrics_fp:
        for sample_idx, batch in enumerate(batches):
            dataset_index = int(args.start_index) + sample_idx
            sample_dir = output_dir / f"sample_{sample_idx:06d}_idx{dataset_index}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            LOG.info("Evaluating sample %d/%d dataset_index=%d", sample_idx + 1, len(batches), dataset_index)

            if (
                bool(getattr(config.model, "unified_vlm_task_prompt", False))
                and batch.get("t5_input_messages")
            ):
                # 与训练侧一致：用统一 task prompt 重建 vlm 消息
                # （见 utils.vlm_utils）；在 meta 提取前替换，记录与输入一致。
                # prompt_mode=text_only 的源用原始文本纯文本编码（不带首帧图）。
                batch = dict(batch)
                if (
                    resolve_prompt_mode(config, batch.get("source_datasets"))
                    == PROMPT_MODE_TEXT_ONLY
                ):
                    batch["vlm_input_messages"] = [
                        str(text).strip() for text in batch["t5_input_messages"]
                    ]
                    batch["first_frame_pils"] = None
                elif bool(getattr(config.model, "vlm_text_rewrite", False)):
                    # vlm_text_rewrite 开启：与训练侧一致改用压缩改写指令消息
                    # （纯文本、不带首帧图；按数据集家族分模板）。
                    source_key = (batch.get("source_datasets") or [None])[0]
                    batch["vlm_input_messages"] = [
                        build_vlm_rewrite_message(text, source_key)
                        for text in batch["t5_input_messages"]
                    ]
                else:
                    batch["vlm_input_messages"] = [
                        build_vlm_task_message(
                            text, style=resolve_vlm_prompt_style(config)
                        )
                        for text in batch["t5_input_messages"]
                    ]

            meta = _sample_meta(batch, sample_idx, dataset_index)
            eval_start = time.time()
            try:
                chunk_info = None
                if args.chunked_teacher_forcing:
                    chunked = _run_chunked_teacher_forcing(
                        model=model,
                        batch=batch,
                        device=device,
                        dtype=dtype,
                        num_inference_steps=int(args.num_inference_steps),
                        chunk_size=int(args.chunk_size),
                        num_chunks=int(args.num_chunks),
                        guidance_scale=guidance_scale,
                        action_inference_params=action_inference_params,
                    )
                    pred_tchw = chunked["pred_tchw"]
                    gt_tchw = chunked["gt_tchw"]
                    conditions_tchw = chunked["conditions_tchw"]
                    metric_pred_tchw = chunked["metric_pred_tchw"]
                    metric_gt_tchw = chunked["metric_gt_tchw"]
                    gt_available = chunked["gt_available"]
                    predicted_actions = chunked["predicted_actions"]
                    chunk_info = chunked["chunks"]
                    first_chw = conditions_tchw[0]
                else:
                    predicted_frames_cthw, predicted_actions = _run_one_sample(
                        model=model,
                        batch=batch,
                        device=device,
                        dtype=dtype,
                        num_inference_steps=int(args.num_inference_steps),
                        guidance_scale=guidance_scale,
                        action_inference_params=action_inference_params,
                    )
                    if predicted_frames_cthw is None:
                        # 视频模态缺失（评审 #4）：inference_step 返回 None，
                        # 本脚本依赖视频输出，明确报错（由外层 except 记录失败）。
                        raise ValueError(
                            "inference_step returned predicted_frames=None "
                            "(video modality missing); eval_video_ckpt requires video"
                        )
                    # model.inference_step returns [B, C, T, H, W]; compare helpers use [T, C, H, W].
                    pred_tchw = predicted_frames_cthw[0].detach().cpu().permute(1, 0, 2, 3).float()
                    gt_tchw = batch["video_frames"][0].detach().cpu().float()
                    first_chw = batch["first_frame"][0].detach().cpu().float()
                    min_t = min(pred_tchw.shape[0], gt_tchw.shape[0])
                    pred_tchw = pred_tchw[:min_t]
                    gt_tchw = gt_tchw[:min_t]
                    conditions_tchw = None
                    metric_pred_tchw = pred_tchw
                    metric_gt_tchw = gt_tchw
                    gt_available = None

                metrics = _compute_video_metrics(metric_pred_tchw, metric_gt_tchw)
                first_uint8 = _image_to_uint8(first_chw)
                gt_uint8 = _tensor_to_uint8_frames(gt_tchw)
                pred_uint8 = _tensor_to_uint8_frames(pred_tchw)
                if conditions_tchw is not None:
                    conditions_uint8 = _tensor_to_uint8_frames(conditions_tchw)
                    compare_uint8 = _make_compare_frames_with_conditions(
                        conditions_uint8,
                        gt_uint8,
                        pred_uint8,
                        include_diff=bool(args.include_diff),
                        gt_available=gt_available,
                    )
                else:
                    conditions_uint8 = None
                    compare_uint8 = _make_compare_frames(
                        first_uint8,
                        gt_uint8,
                        pred_uint8,
                        include_diff=bool(args.include_diff),
                    )

                Image.fromarray(first_uint8).save(sample_dir / "condition.png")
                _make_grid(first_uint8, gt_uint8, pred_uint8).save(sample_dir / "grid.png")
                _write_video(sample_dir / "gt.mp4", gt_uint8, fps=int(args.fps))
                _write_video(sample_dir / "pred.mp4", pred_uint8, fps=int(args.fps))
                _write_video(sample_dir / "compare.mp4", compare_uint8, fps=int(args.fps))
                if conditions_uint8 is not None:
                    _write_video(sample_dir / "conditions.mp4", conditions_uint8, fps=int(args.fps))

                if predicted_actions is not None and int(predicted_actions.numel()) > 0:
                    torch.save(predicted_actions.detach().cpu(), sample_dir / "predicted_actions.pt")
                record = {
                    **meta,
                    "ok": True,
                    "seconds": time.time() - eval_start,
                    "num_inference_steps": int(args.num_inference_steps),
                    "chunked_teacher_forcing": bool(args.chunked_teacher_forcing),
                    "chunk_size": int(args.chunk_size) if args.chunked_teacher_forcing else None,
                    "num_chunks": int(args.num_chunks) if args.chunked_teacher_forcing else None,
                    "chunks": chunk_info,
                    "output_dir": str(sample_dir),
                    **metrics,
                }
            except Exception as exc:
                LOG.exception("Sample failed: dataset_index=%d", dataset_index)
                record = {
                    **meta,
                    "ok": False,
                    "seconds": time.time() - eval_start,
                    "error": repr(exc),
                    "output_dir": str(sample_dir),
                }

            _json_dump(sample_dir / "meta.json", record)
            metrics_fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            metrics_fp.flush()
            records.append(record)

            del batch
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()

    ok_records = [record for record in records if record.get("ok")]
    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": "pretrained_backbones_only" if args.skip_checkpoint else str(Path(args.checkpoint).resolve()),
        "output": str(output_dir.resolve()),
        "device": str(device),
        "dtype": args.dtype,
        "num_samples_requested": int(args.num_samples),
        "num_samples_selected": len(indices),
        "num_samples_ok": len(ok_records),
        "num_samples_failed": len(records) - len(ok_records),
        "indices": [int(args.start_index) + i for i in range(len(batches))],
        "load_info": load_info,
        "elapsed_seconds": time.time() - start_time,
    }
    for key in ("mse", "mae", "psnr", "ssim"):
        vals = [record[key] for record in ok_records if record.get(key) is not None]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))
    _json_dump(output_dir / "summary.json", summary)
    LOG.info("Done. Summary written to %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
